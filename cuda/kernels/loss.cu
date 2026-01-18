// Loss functions: CrossEntropy, MoE Auxiliary Loss
#include "common.cuh"

// Cross Entropy Loss

// Fused log_softmax + NLL loss forward
// logits: [B*T, V], targets: [B*T], loss: scalar
__global__ void cross_entropy_forward_kernel(const float* __restrict__ logits,
                                             const int* __restrict__ targets,
                                             float* __restrict__ loss,
                                             float* __restrict__ log_probs,
                                             int batch_seq,
                                             int vocab_size) {
    int idx = blockIdx.x;
    if (idx >= batch_seq) return;

    const float* row = logits + idx * vocab_size;
    float* log_row = log_probs + idx * vocab_size;
    int target = targets[idx];

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        max_val = fmaxf(max_val, row[v]);
    }
    max_val = warp_reduce_max(max_val);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;

    // Compute sum(exp(x - max))
    float sum_exp = 0.0f;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        sum_exp += expf(row[v] - max_val);
    }
    sum_exp = block_reduce_sum(sum_exp);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();
    float log_sum_exp = logf(s_sum);

    // Compute log_softmax and store
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        log_row[v] = row[v] - max_val - log_sum_exp;
    }

    // NLL loss for this sample
    if (threadIdx.x == 0) {
        float nll = -(row[target] - max_val - log_sum_exp);
        atomicAdd(loss, nll / batch_seq);
    }
}

// Cross entropy backward: d_logits = softmax(logits) - one_hot(target)
__global__ void cross_entropy_backward_kernel(const float* __restrict__ log_probs,
                                              const int* __restrict__ targets,
                                              float* __restrict__ d_logits,
                                              int batch_seq,
                                              int vocab_size,
                                              float scale) {
    int idx = blockIdx.x;
    if (idx >= batch_seq) return;

    const float* log_row = log_probs + idx * vocab_size;
    float* d_row = d_logits + idx * vocab_size;
    int target = targets[idx];

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float prob = expf(log_row[v]);
        float grad = prob - (v == target ? 1.0f : 0.0f);
        d_row[v] = grad * scale;
    }
}

// =============================================================================
// MoE Auxiliary Loss (Load Balancing)
// =============================================================================

// Compute auxiliary loss for MoE load balancing
// router_probs: [B*T, n_experts], expert_indices: [B*T, top_k]
// L_aux = α * Σ_i (f_i * P_i)
__global__ void aux_loss_forward_kernel(const float* __restrict__ router_probs,
                                        const int* __restrict__ expert_indices,
                                        float* __restrict__ f_counts,
                                        float* __restrict__ p_means,
                                        int batch_seq,
                                        int n_experts,
                                        int top_k) {
    // Phase 1: Count tokens per expert and accumulate probabilities
    int expert = blockIdx.x;
    if (expert >= n_experts) return;

    float count = 0.0f;
    float prob_sum = 0.0f;

    for (int i = threadIdx.x; i < batch_seq; i += blockDim.x) {
        // Check if this expert was selected for token i
        for (int k = 0; k < top_k; k++) {
            if (expert_indices[i * top_k + k] == expert) {
                count += 1.0f;
                break;
            }
        }
        prob_sum += router_probs[i * n_experts + expert];
    }

    count = block_reduce_sum(count);
    prob_sum = block_reduce_sum(prob_sum);

    if (threadIdx.x == 0) {
        f_counts[expert] = count / batch_seq;
        p_means[expert] = prob_sum / batch_seq;
    }
}

__global__ void aux_loss_reduce_kernel(const float* __restrict__ f_counts,
                                       const float* __restrict__ p_means,
                                       float* __restrict__ aux_loss,
                                       int n_experts,
                                       float alpha) {
    float loss = 0.0f;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
        loss += f_counts[i] * p_means[i];
    }
    loss = block_reduce_sum(loss);

    if (threadIdx.x == 0) {
        *aux_loss = alpha * n_experts * loss;
    }
}

extern "C" {

int32_t cuda_cross_entropy_forward(const float* logits, const int* targets,
                                   float* loss, float* log_probs,
                                   int batch_seq, int vocab_size, void* stream) {
    // Zero loss
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), s));

    int threads = min(vocab_size, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    cross_entropy_forward_kernel<<<batch_seq, threads, 0, s>>>(
        logits, targets, loss, log_probs, batch_seq, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_cross_entropy_backward(const float* log_probs, const int* targets,
                                    float* d_logits, int batch_seq, int vocab_size,
                                    float scale, void* stream) {
    int threads = min(vocab_size, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cross_entropy_backward_kernel<<<batch_seq, threads, 0, s>>>(
        log_probs, targets, d_logits, batch_seq, vocab_size, scale);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_aux_loss_forward(const float* router_probs, const int* expert_indices,
                              float* aux_loss, float* f_counts, float* p_means,
                              int batch_seq, int n_experts, int top_k,
                              float alpha, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    int threads = min(batch_seq, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    aux_loss_forward_kernel<<<n_experts, threads, 0, s>>>(
        router_probs, expert_indices, f_counts, p_means, batch_seq, n_experts, top_k);
    CUDA_CHECK(cudaGetLastError());

    aux_loss_reduce_kernel<<<1, WARP_SIZE, 0, s>>>(f_counts, p_means, aux_loss, n_experts, alpha);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
