// RoPE (Rotary Position Embedding) kernel with NTK scaling
#include "common.cuh"
#include <cmath>

// Precompute RoPE frequencies with NTK scaling
// freqs[pos, d/2] where d is head_dim
__global__ void rope_freqs_kernel(float* __restrict__ freqs,
                                  int max_seq_len,
                                  int head_dim,
                                  float base,
                                  float alpha) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos >= max_seq_len || d >= head_dim / 2) return;

    // NTK-aware base scaling
    float ntk_base = base * powf(alpha, (float)head_dim / (head_dim - 2));
    float freq = 1.0f / powf(ntk_base, (float)(2 * d) / head_dim);
    float angle = pos * freq;

    int idx = pos * (head_dim / 2) + d;
    freqs[idx] = angle;
}

// Apply RoPE to Q/K tensors
// input: [batch, seq_len, n_heads, head_dim]
// freqs: [max_seq_len, head_dim/2] (precomputed angles)
__global__ void rope_forward_kernel(const float* __restrict__ input,
                                    const float* __restrict__ freqs,
                                    float* __restrict__ output,
                                    int batch,
                                    int seq_len,
                                    int n_heads,
                                    int head_dim,
                                    int offset) {  // position offset for KV cache
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;

    if (b >= batch || s >= seq_len || h >= n_heads || d >= head_dim / 2) return;

    int pos = s + offset;
    float angle = freqs[pos * (head_dim / 2) + d];
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int base_idx = ((b * seq_len + s) * n_heads + h) * head_dim;

    // x[..., 0::2] and x[..., 1::2]
    float x0 = input[base_idx + 2 * d];
    float x1 = input[base_idx + 2 * d + 1];

    // Apply rotation
    output[base_idx + 2 * d] = x0 * cos_val - x1 * sin_val;
    output[base_idx + 2 * d + 1] = x0 * sin_val + x1 * cos_val;
}

// Fused QKV projection + RoPE for efficiency
__global__ void rope_qk_kernel(const float* __restrict__ q,
                               const float* __restrict__ k,
                               const float* __restrict__ freqs,
                               float* __restrict__ q_out,
                               float* __restrict__ k_out,
                               int batch,
                               int seq_len,
                               int n_q_heads,
                               int n_kv_heads,
                               int head_dim,
                               int offset) {
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = threadIdx.x;

    if (b >= batch || s >= seq_len || d >= head_dim / 2) return;

    int pos = s + offset;
    float angle = freqs[pos * (head_dim / 2) + d];
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Apply to all Q heads
    for (int h = 0; h < n_q_heads; h++) {
        int q_idx = ((b * seq_len + s) * n_q_heads + h) * head_dim;
        float q0 = q[q_idx + 2 * d];
        float q1 = q[q_idx + 2 * d + 1];
        q_out[q_idx + 2 * d] = q0 * cos_val - q1 * sin_val;
        q_out[q_idx + 2 * d + 1] = q0 * sin_val + q1 * cos_val;
    }

    // Apply to all KV heads
    for (int h = 0; h < n_kv_heads; h++) {
        int k_idx = ((b * seq_len + s) * n_kv_heads + h) * head_dim;
        float k0 = k[k_idx + 2 * d];
        float k1 = k[k_idx + 2 * d + 1];
        k_out[k_idx + 2 * d] = k0 * cos_val - k1 * sin_val;
        k_out[k_idx + 2 * d + 1] = k0 * sin_val + k1 * cos_val;
    }
}

extern "C" {

int32_t cuda_rope_freqs(float* freqs, int max_seq_len, int head_dim,
                        float base, float alpha, void* stream) {
    dim3 threads(32, 16);
    dim3 blocks((max_seq_len + 31) / 32, (head_dim / 2 + 15) / 16);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rope_freqs_kernel<<<blocks, threads, 0, s>>>(freqs, max_seq_len, head_dim, base, alpha);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_rope_forward(const float* input, const float* freqs, float* output,
                          int batch, int seq_len, int n_heads, int head_dim,
                          int offset, void* stream) {
    dim3 threads(head_dim / 2);
    dim3 blocks(n_heads, seq_len, batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rope_forward_kernel<<<blocks, threads, 0, s>>>(
        input, freqs, output, batch, seq_len, n_heads, head_dim, offset);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_rope_qk(const float* q, const float* k, const float* freqs,
                     float* q_out, float* k_out,
                     int batch, int seq_len, int n_q_heads, int n_kv_heads,
                     int head_dim, int offset, void* stream) {
    dim3 threads(head_dim / 2);
    dim3 blocks(1, seq_len, batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rope_qk_kernel<<<blocks, threads, 0, s>>>(
        q, k, freqs, q_out, k_out, batch, seq_len, n_q_heads, n_kv_heads, head_dim, offset);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
