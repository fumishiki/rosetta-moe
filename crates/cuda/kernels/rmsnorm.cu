// RMSNorm kernel
#include "common.cuh"

// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
// Each block handles one row (hidden_dim elements)
__global__ void rmsnorm_kernel(const float* __restrict__ input,
                               const float* __restrict__ weight,
                               float* __restrict__ output,
                               int rows,
                               int hidden_dim,
                               float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * hidden_dim;
    float* row_out = output + row * hidden_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = row_in[i];
        sum_sq += val * val;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();
    float rms_inv = s_rms;

    // Apply normalization
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

// Fused RMSNorm + residual add
__global__ void rmsnorm_residual_kernel(const float* __restrict__ input,
                                        const float* __restrict__ residual,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        float* __restrict__ residual_out,
                                        int rows,
                                        int hidden_dim,
                                        float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * hidden_dim;
    const float* row_res = residual + row * hidden_dim;
    float* row_out = output + row * hidden_dim;
    float* row_res_out = residual_out + row * hidden_dim;

    extern __shared__ float smem[];

    // Add residual and compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = row_in[i] + row_res[i];
        smem[i] = val;
        sum_sq += val * val;
    }
    __syncthreads();

    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();
    float rms_inv = s_rms;

    // Apply normalization and save residual
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = smem[i];
        row_res_out[i] = val;
        row_out[i] = val * rms_inv * weight[i];
    }
}

extern "C" {

int32_t cuda_rmsnorm(const float* input, const float* weight, float* output,
                     int rows, int hidden_dim, float eps, void* stream) {
    int threads = min(hidden_dim, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_kernel<<<rows, threads, 0, s>>>(input, weight, output, rows, hidden_dim, eps);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_rmsnorm_residual(const float* input, const float* residual, const float* weight,
                              float* output, float* residual_out,
                              int rows, int hidden_dim, float eps, void* stream) {
    int threads = min(hidden_dim, MAX_THREADS_PER_BLOCK);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    size_t smem_size = hidden_dim * sizeof(float);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_residual_kernel<<<rows, threads, smem_size, s>>>(
        input, residual, weight, output, residual_out, rows, hidden_dim, eps);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
