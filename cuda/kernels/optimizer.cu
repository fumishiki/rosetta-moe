// Optimizer kernels: AdamW, Gradient Clipping
#include "common.cuh"

// AdamW Optimizer
// Fused AdamW step
// p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
__global__ void adamw_step_kernel(float* __restrict__ param,
                                  const float* __restrict__ grad,
                                  float* __restrict__ m,
                                  float* __restrict__ v,
                                  float lr,
                                  float beta1,
                                  float beta2,
                                  float eps,
                                  float weight_decay,
                                  float bias_correction1,
                                  float bias_correction2,
                                  int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx];
    float m_old = m[idx];
    float v_old = v[idx];

    // Update biased first moment
    float m_new = beta1 * m_old + (1.0f - beta1) * g;
    // Update biased second moment
    float v_new = beta2 * v_old + (1.0f - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;

    // AdamW update (decoupled weight decay)
    float p = param[idx];
    p = p - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
    param[idx] = p;
}

// Zero gradients
__global__ void zero_grad_kernel(float* __restrict__ grad, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    grad[idx] = 0.0f;
}

// =============================================================================
// Gradient Clipping
// =============================================================================

// Compute global L2 norm squared (partial sum per block)
__global__ void grad_norm_kernel(const float* __restrict__ grad,
                                 float* __restrict__ partial_norms,
                                 int64_t size) {
    extern __shared__ float smem[];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (idx < size) {
        float g = grad[idx];
        sum = g * g;
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        partial_norms[blockIdx.x] = sum;
    }
}

// Reduce partial norms to final norm
__global__ void grad_norm_reduce_kernel(const float* __restrict__ partial_norms,
                                        float* __restrict__ total_norm,
                                        int num_blocks) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += partial_norms[i];
    }
    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        *total_norm = sqrtf(sum);
    }
}

// Clip gradients by global norm
__global__ void grad_clip_kernel(float* __restrict__ grad,
                                 float clip_norm,
                                 float total_norm,
                                 int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (total_norm > clip_norm) {
        float scale = clip_norm / total_norm;
        grad[idx] *= scale;
    }
}

// =============================================================================
// Scatter Add (for Embedding backward)
// =============================================================================

// Atomic scatter add: grad_weight[indices[i]] += grad_output[i]
__global__ void scatter_add_kernel(const float* __restrict__ grad_output,
                                   const int* __restrict__ indices,
                                   float* __restrict__ grad_weight,
                                   int num_indices,
                                   int embedding_dim) {
    int i = blockIdx.x;
    int d = threadIdx.x;

    if (i >= num_indices || d >= embedding_dim) return;

    int idx = indices[i];
    float grad = grad_output[i * embedding_dim + d];
    atomicAdd(&grad_weight[idx * embedding_dim + d], grad);
}

extern "C" {

int32_t cuda_adamw_step(float* param, const float* grad, float* m, float* v,
                        float lr, float beta1, float beta2, float eps,
                        float weight_decay, int step, int64_t size, void* stream) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    adamw_step_kernel<<<blocks, threads, 0, s>>>(
        param, grad, m, v, lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_zero_grad(float* grad, int64_t size, void* stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    zero_grad_kernel<<<blocks, threads, 0, s>>>(grad, size);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_grad_clip(float* grad, float* partial_norms, float* total_norm,
                       float clip_norm, int64_t size, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Compute partial norms
    grad_norm_kernel<<<blocks, threads, 0, s>>>(grad, partial_norms, size);
    CUDA_CHECK(cudaGetLastError());

    // Reduce to total norm
    grad_norm_reduce_kernel<<<1, min(blocks, 1024), 0, s>>>(partial_norms, total_norm, blocks);
    CUDA_CHECK(cudaGetLastError());

    // Clip
    grad_clip_kernel<<<blocks, threads, 0, s>>>(grad, clip_norm, *total_norm, size);
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

int32_t cuda_scatter_add(const float* grad_output, const int* indices,
                         float* grad_weight, int num_indices, int embedding_dim,
                         void* stream) {
    dim3 threads(min(embedding_dim, 1024));
    dim3 blocks(num_indices);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    scatter_add_kernel<<<blocks, threads, 0, s>>>(
        grad_output, indices, grad_weight, num_indices, embedding_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
