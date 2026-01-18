// GEMM (General Matrix Multiply) kernel
// C = A @ B^T (for linear layer: output = input @ weight.T)
#include "common.cuh"

// Tile size for shared memory tiling
constexpr int TILE_SIZE = 32;

// Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T
// B is stored as [N,K] (row-major), so we compute A @ B^T
__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile (B is [N,K], we want B^T, so B[col, k])
        int b_col = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// GEMM with beta accumulation: C = alpha * A @ B^T + beta * C
__global__ void gemm_beta_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K,
                                 float alpha, float beta) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_col = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

// Batched GEMM for attention: C[b,M,N] = A[b,M,K] @ B[b,N,K]^T
__global__ void batched_gemm_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int batch, int M, int N, int K) {
    int b = blockIdx.z;
    if (b >= batch) return;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * N * K;
    float* C_batch = C + b * M * N;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A_batch[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_col = t * TILE_SIZE + threadIdx.y;
        if (col < N && b_col < K) {
            Bs[threadIdx.y][threadIdx.x] = B_batch[col * K + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

extern "C" {

int32_t cuda_gemm(const float* A, const float* B, float* C,
                  int M, int N, int K, void* stream) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    gemm_kernel<<<blocks, threads, 0, s>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_gemm_beta(const float* A, const float* B, float* C,
                       int M, int N, int K, float alpha, float beta, void* stream) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    gemm_beta_kernel<<<blocks, threads, 0, s>>>(A, B, C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_batched_gemm(const float* A, const float* B, float* C,
                          int batch, int M, int N, int K, void* stream) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE,
                batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    batched_gemm_kernel<<<blocks, threads, 0, s>>>(A, B, C, batch, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
