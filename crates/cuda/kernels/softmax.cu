// Softmax kernel (online softmax for numerical stability)
#include "common.cuh"

// Row-wise softmax: each block handles one row
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int rows,
                               int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    // Phase 1: Find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    max_val = warp_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Phase 2: Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(row_in[i] - max_val);
        row_out[i] = exp_val;
        sum += exp_val;
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// Fused softmax + top-k for router
__global__ void softmax_topk_kernel(const float* __restrict__ input,
                                    float* __restrict__ weights,
                                    int32_t* __restrict__ indices,
                                    int rows,
                                    int cols,
                                    int k) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* s_vals = smem;
    int* s_idx = reinterpret_cast<int*>(smem + cols);

    const float* row_in = input + row * cols;
    float* row_weights = weights + row * k;
    int32_t* row_indices = indices + row * k;

    // Load and find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = row_in[i];
        s_vals[i] = v;
        s_idx[i] = i;
        max_val = fmaxf(max_val, v);
    }
    __syncthreads();

    max_val = warp_reduce_max(max_val);
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Compute softmax
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(s_vals[i] - max_val);
        s_vals[i] = exp_val;
        sum += exp_val;
    }
    sum = block_reduce_sum(sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();

    float inv_sum = 1.0f / s_sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        s_vals[i] *= inv_sum;
    }
    __syncthreads();

    // Simple top-k selection (bubble sort for small k)
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            int max_idx = i;
            float max_v = s_vals[i];
            for (int j = i + 1; j < cols; j++) {
                if (s_vals[j] > max_v) {
                    max_v = s_vals[j];
                    max_idx = j;
                }
            }
            // Swap
            float tmp_v = s_vals[i];
            int tmp_i = s_idx[i];
            s_vals[i] = s_vals[max_idx];
            s_idx[i] = s_idx[max_idx];
            s_vals[max_idx] = tmp_v;
            s_idx[max_idx] = tmp_i;

            row_weights[i] = s_vals[i];
            row_indices[i] = s_idx[i];
        }

        // Renormalize top-k weights
        float topk_sum = 0.0f;
        for (int i = 0; i < k; i++) {
            topk_sum += row_weights[i];
        }
        float inv_topk = 1.0f / topk_sum;
        for (int i = 0; i < k; i++) {
            row_weights[i] *= inv_topk;
        }
    }
}

extern "C" {

int32_t cuda_softmax(const float* input, float* output, int rows, int cols, void* stream) {
    int threads = min(cols, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    softmax_kernel<<<rows, threads, 0, s>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_softmax_topk(const float* input, float* weights, int32_t* indices,
                          int rows, int cols, int k, void* stream) {
    int threads = min(cols, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    size_t smem_size = cols * sizeof(float) + cols * sizeof(int);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    softmax_topk_kernel<<<rows, threads, smem_size, s>>>(input, weights, indices, rows, cols, k);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
