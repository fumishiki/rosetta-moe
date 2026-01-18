// Element-wise operations: SiLU, Add, Mul
#include "common.cuh"

// SiLU activation: x * sigmoid(x)
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// Element-wise add
__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ output,
                           int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}

// Element-wise multiply
__global__ void mul_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ output,
                           int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx];
    }
}

// Scale (multiply by scalar)
__global__ void scale_kernel(const float* __restrict__ input,
                             float scale,
                             float* __restrict__ output,
                             int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scale;
    }
}

extern "C" {

int32_t cuda_silu(const float* input, float* output, int64_t n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    silu_kernel<<<blocks, threads, 0, s>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_add(const float* a, const float* b, float* output, int64_t n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    add_kernel<<<blocks, threads, 0, s>>>(a, b, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_mul(const float* a, const float* b, float* output, int64_t n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    mul_kernel<<<blocks, threads, 0, s>>>(a, b, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_scale(const float* input, float scale, float* output, int64_t n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    scale_kernel<<<blocks, threads, 0, s>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
