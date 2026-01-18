// Decoding kernels: argmax, sampling (GPU-side token generation)
#include "common.cuh"

// Argmax (Greedy Decoding)
// Per-row argmax: find index of maximum value
// logits: [batch, vocab_size], output: [batch] (int32)
__global__ void argmax_kernel(const float* __restrict__ logits,
                              int32_t* __restrict__ output,
                              int batch,
                              int vocab_size) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float* row_logits = logits + row * vocab_size;

    // Each thread finds local max
    float max_val = -INFINITY;
    int max_idx = 0;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = row_logits[v];
        if (val > max_val) {
            max_val = val;
            max_idx = v;
        }
    }

    // Warp-level reduction to find global max
    __shared__ float s_max_vals[32];
    __shared__ int s_max_idxs[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    if (lane == 0) {
        s_max_vals[wid] = max_val;
        s_max_idxs[wid] = max_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (wid == 0) {
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        max_val = (lane < num_warps) ? s_max_vals[lane] : -INFINITY;
        max_idx = (lane < num_warps) ? s_max_idxs[lane] : 0;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
            if (other_val > max_val) {
                max_val = other_val;
                max_idx = other_idx;
            }
        }

        if (lane == 0) {
            output[row] = max_idx;
        }
    }
}

// Multinomial Sampling with Temperature
// Simple LCG random number generator (per-thread state)
__device__ __forceinline__ uint32_t lcg_random(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

__device__ __forceinline__ float lcg_uniform(uint32_t* state) {
    return static_cast<float>(lcg_random(state)) / 4294967296.0f;
}

// Sample from logits with temperature
// logits: [batch, vocab_size], output: [batch] (int32)
// seeds: [batch] random seeds for reproducibility
__global__ void sample_kernel(const float* __restrict__ logits,
                              int32_t* __restrict__ output,
                              const uint64_t* __restrict__ seeds,
                              int batch,
                              int vocab_size,
                              float temperature) {
    int row = blockIdx.x;
    if (row >= batch) return;

    const float* row_logits = logits + row * vocab_size;
    uint32_t rng_state = static_cast<uint32_t>(seeds[row] ^ (row * 12345));

    // Apply temperature and compute softmax
    extern __shared__ float smem[];
    float* probs = smem;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        max_val = fmaxf(max_val, row_logits[v]);
    }
    max_val = warp_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Compute exp((logit - max) / temperature)
    float sum = 0.0f;
    float inv_temp = 1.0f / temperature;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float p = expf((row_logits[v] - max_val) * inv_temp);
        probs[v] = p;
        sum += p;
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();

    // Normalize to probabilities
    float inv_sum = 1.0f / s_sum;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        probs[v] *= inv_sum;
    }
    __syncthreads();

    // Sample using cumulative distribution (single thread)
    if (threadIdx.x == 0) {
        float u = lcg_uniform(&rng_state);
        float cumsum = 0.0f;
        int sampled = vocab_size - 1;

        for (int v = 0; v < vocab_size; v++) {
            cumsum += probs[v];
            if (u < cumsum) {
                sampled = v;
                break;
            }
        }
        output[row] = sampled;
    }
}

// =============================================================================
// Top-K Sampling
// =============================================================================

// Sample from top-k logits with temperature
__global__ void topk_sample_kernel(const float* __restrict__ logits,
                                   int32_t* __restrict__ output,
                                   const uint64_t* __restrict__ seeds,
                                   int batch,
                                   int vocab_size,
                                   int k,
                                   float temperature) {
    int row = blockIdx.x;
    if (row >= batch) return;

    extern __shared__ char shared_mem[];
    float* s_vals = reinterpret_cast<float*>(shared_mem);
    int* s_idxs = reinterpret_cast<int*>(shared_mem + k * sizeof(float));

    const float* row_logits = logits + row * vocab_size;
    uint32_t rng_state = static_cast<uint32_t>(seeds[row] ^ (row * 12345));

    // Find top-k (simple selection, thread 0 only for simplicity)
    if (threadIdx.x == 0) {
        // Initialize with -inf
        for (int i = 0; i < k; i++) {
            s_vals[i] = -INFINITY;
            s_idxs[i] = 0;
        }

        // Find top-k
        for (int v = 0; v < vocab_size; v++) {
            float val = row_logits[v];
            // Check if this value should be in top-k
            if (val > s_vals[k - 1]) {
                // Insert in sorted position
                int pos = k - 1;
                while (pos > 0 && val > s_vals[pos - 1]) {
                    s_vals[pos] = s_vals[pos - 1];
                    s_idxs[pos] = s_idxs[pos - 1];
                    pos--;
                }
                s_vals[pos] = val;
                s_idxs[pos] = v;
            }
        }

        // Apply temperature and compute softmax over top-k
        float max_val = s_vals[0];
        float inv_temp = 1.0f / temperature;
        float sum = 0.0f;

        for (int i = 0; i < k; i++) {
            float p = expf((s_vals[i] - max_val) * inv_temp);
            s_vals[i] = p;
            sum += p;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < k; i++) {
            s_vals[i] *= inv_sum;
        }

        // Sample
        float u = lcg_uniform(&rng_state);
        float cumsum = 0.0f;
        int sampled_idx = k - 1;

        for (int i = 0; i < k; i++) {
            cumsum += s_vals[i];
            if (u < cumsum) {
                sampled_idx = i;
                break;
            }
        }

        output[row] = s_idxs[sampled_idx];
    }
}

// =============================================================================
// Top-P (Nucleus) Sampling
// =============================================================================

// Sample from nucleus (top-p) with temperature
__global__ void topp_sample_kernel(const float* __restrict__ logits,
                                   int32_t* __restrict__ output,
                                   const uint64_t* __restrict__ seeds,
                                   int batch,
                                   int vocab_size,
                                   float top_p,
                                   float temperature) {
    int row = blockIdx.x;
    if (row >= batch) return;

    extern __shared__ char shared_mem[];
    float* s_probs = reinterpret_cast<float*>(shared_mem);
    int* s_idxs = reinterpret_cast<int*>(shared_mem + vocab_size * sizeof(float));

    const float* row_logits = logits + row * vocab_size;
    uint32_t rng_state = static_cast<uint32_t>(seeds[row] ^ (row * 12345));

    // Thread 0 does all work (simple implementation for correctness)
    if (threadIdx.x == 0) {
        // Apply temperature and compute softmax
        float max_val = -INFINITY;
        for (int v = 0; v < vocab_size; v++) {
            max_val = fmaxf(max_val, row_logits[v]);
        }

        float inv_temp = 1.0f / temperature;
        float sum = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            float p = expf((row_logits[v] - max_val) * inv_temp);
            s_probs[v] = p;
            s_idxs[v] = v;
            sum += p;
        }

        float inv_sum = 1.0f / sum;
        for (int v = 0; v < vocab_size; v++) {
            s_probs[v] *= inv_sum;
        }

        // Sort by probability (descending) - simple insertion sort
        for (int i = 1; i < vocab_size; i++) {
            float p = s_probs[i];
            int idx = s_idxs[i];
            int j = i - 1;
            while (j >= 0 && s_probs[j] < p) {
                s_probs[j + 1] = s_probs[j];
                s_idxs[j + 1] = s_idxs[j];
                j--;
            }
            s_probs[j + 1] = p;
            s_idxs[j + 1] = idx;
        }

        // Find nucleus (cumsum until top_p)
        float cumsum = 0.0f;
        int nucleus_size = vocab_size;
        for (int v = 0; v < vocab_size; v++) {
            cumsum += s_probs[v];
            if (cumsum >= top_p) {
                nucleus_size = v + 1;
                break;
            }
        }

        // Renormalize nucleus
        float nucleus_sum = 0.0f;
        for (int v = 0; v < nucleus_size; v++) {
            nucleus_sum += s_probs[v];
        }
        float inv_nucleus = 1.0f / nucleus_sum;

        // Sample from nucleus
        float u = lcg_uniform(&rng_state);
        cumsum = 0.0f;
        int sampled_idx = nucleus_size - 1;

        for (int v = 0; v < nucleus_size; v++) {
            cumsum += s_probs[v] * inv_nucleus;
            if (u < cumsum) {
                sampled_idx = v;
                break;
            }
        }

        output[row] = s_idxs[sampled_idx];
    }
}

// C Interface
extern "C" {

int32_t cuda_argmax(const float* logits, int32_t* output,
                    int batch, int vocab_size, void* stream) {
    int threads = min(vocab_size, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    argmax_kernel<<<batch, threads, 0, s>>>(logits, output, batch, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_sample(const float* logits, int32_t* output,
                    const uint64_t* seeds, int batch, int vocab_size,
                    float temperature, void* stream) {
    int threads = min(vocab_size, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    size_t smem_size = vocab_size * sizeof(float);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    sample_kernel<<<batch, threads, smem_size, s>>>(
        logits, output, seeds, batch, vocab_size, temperature);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_topk_sample(const float* logits, int32_t* output,
                         const uint64_t* seeds, int batch, int vocab_size,
                         int k, float temperature, void* stream) {
    size_t smem_size = k * (sizeof(float) + sizeof(int));
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    topk_sample_kernel<<<batch, 1, smem_size, s>>>(
        logits, output, seeds, batch, vocab_size, k, temperature);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_topp_sample(const float* logits, int32_t* output,
                         const uint64_t* seeds, int batch, int vocab_size,
                         float top_p, float temperature, void* stream) {
    size_t smem_size = vocab_size * (sizeof(float) + sizeof(int));
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    topp_sample_kernel<<<batch, 1, smem_size, s>>>(
        logits, output, seeds, batch, vocab_size, top_p, temperature);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
