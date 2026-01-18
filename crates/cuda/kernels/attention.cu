// Multi-Query Attention kernel
// Fused attention with FlashAttention-style memory optimization
#include "common.cuh"

// Simple attention (for reference/fallback)
// scores = Q @ K^T / sqrt(head_dim)
// output = softmax(scores) @ V
__global__ void attention_scores_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        float* __restrict__ scores,
                                        int batch,
                                        int n_heads,
                                        int seq_len,
                                        int head_dim,
                                        float scale) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch || h >= n_heads || q_pos >= seq_len) return;

    // Q[b, q_pos, h, :] dot K[b, k_pos, 0, :] for MQA (single KV head)
    const float* q_ptr = Q + ((b * seq_len + q_pos) * n_heads + h) * head_dim;
    float* score_row = scores + ((b * n_heads + h) * seq_len + q_pos) * seq_len;

    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        // MQA: K has only 1 head, so ignore h for K index
        const float* k_ptr = K + (b * seq_len + k_pos) * head_dim;

        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_ptr[d] * k_ptr[d];
        }

        // Causal mask: only attend to positions <= q_pos
        if (k_pos <= q_pos) {
            score_row[k_pos] = dot * scale;
        } else {
            score_row[k_pos] = -INFINITY;
        }
    }
}

// Apply attention output: output = attention_weights @ V
__global__ void attention_output_kernel(const float* __restrict__ weights,
                                        const float* __restrict__ V,
                                        float* __restrict__ output,
                                        int batch,
                                        int n_heads,
                                        int seq_len,
                                        int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x;
    int d = threadIdx.x;

    if (b >= batch || h >= n_heads || q_pos >= seq_len || d >= head_dim) return;

    const float* w_row = weights + ((b * n_heads + h) * seq_len + q_pos) * seq_len;
    float* out_ptr = output + ((b * seq_len + q_pos) * n_heads + h) * head_dim + d;

    float sum = 0.0f;
    for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
        // MQA: V has only 1 head
        const float* v_ptr = V + (b * seq_len + k_pos) * head_dim + d;
        sum += w_row[k_pos] * (*v_ptr);
    }

    *out_ptr = sum;
}

// FlashAttention-style fused kernel (simplified version)
// Processes attention in blocks to minimize memory usage
__global__ void flash_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ output,
                                       int batch,
                                       int n_q_heads,
                                       int n_kv_heads,
                                       int seq_len,
                                       int head_dim,
                                       float scale) {
    extern __shared__ float smem[];

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_pos = blockIdx.x;
    int d = threadIdx.x;

    if (b >= batch || h >= n_q_heads || q_pos >= seq_len) return;

    // Shared memory layout: [K block | V block | scores]
    constexpr int BLOCK_SIZE = 64;  // Process 64 K/V positions at a time
    float* s_k = smem;
    float* s_v = smem + BLOCK_SIZE * head_dim;
    float* s_scores = smem + 2 * BLOCK_SIZE * head_dim;

    // KV head index (for MQA/GQA)
    int kv_h = h * n_kv_heads / n_q_heads;

    // Load Q
    const float* q_ptr = Q + ((b * seq_len + q_pos) * n_q_heads + h) * head_dim;
    float q_val = (d < head_dim) ? q_ptr[d] : 0.0f;

    // Online softmax variables
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o_acc = 0.0f;

    // Process K/V in blocks
    for (int block_start = 0; block_start <= q_pos; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, q_pos + 1);
        int block_len = block_end - block_start;

        // Load K and V block to shared memory
        __syncthreads();
        for (int i = threadIdx.x; i < block_len * head_dim; i += blockDim.x) {
            int k_pos = block_start + i / head_dim;
            int k_d = i % head_dim;
            if (k_pos <= q_pos) {
                s_k[i] = K[((b * seq_len + k_pos) * n_kv_heads + kv_h) * head_dim + k_d];
                s_v[i] = V[((b * seq_len + k_pos) * n_kv_heads + kv_h) * head_dim + k_d];
            }
        }
        __syncthreads();

        // Compute scores for this block
        if (d == 0) {
            for (int i = 0; i < block_len; i++) {
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    dot += q_ptr[dd] * s_k[i * head_dim + dd];
                }
                s_scores[i] = dot * scale;
            }
        }
        __syncthreads();

        // Online softmax update
        float m_cur = m_prev;
        for (int i = 0; i < block_len; i++) {
            m_cur = fmaxf(m_cur, s_scores[i]);
        }

        float l_cur = l_prev * expf(m_prev - m_cur);
        float o_scale = expf(m_prev - m_cur);
        o_acc *= o_scale;

        for (int i = 0; i < block_len; i++) {
            float p = expf(s_scores[i] - m_cur);
            l_cur += p;
            if (d < head_dim) {
                o_acc += p * s_v[i * head_dim + d];
            }
        }

        m_prev = m_cur;
        l_prev = l_cur;
    }

    // Finalize output
    if (d < head_dim) {
        float* out_ptr = output + ((b * seq_len + q_pos) * n_q_heads + h) * head_dim + d;
        *out_ptr = o_acc / l_prev;
    }
}

extern "C" {

int32_t cuda_attention_scores(const float* Q, const float* K, float* scores,
                              int batch, int n_heads, int seq_len, int head_dim,
                              float scale, void* stream) {
    dim3 threads(min(seq_len, 256));
    dim3 blocks((seq_len + 255) / 256, n_heads, batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    attention_scores_kernel<<<blocks, threads, 0, s>>>(
        Q, K, scores, batch, n_heads, seq_len, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_attention_output(const float* weights, const float* V, float* output,
                              int batch, int n_heads, int seq_len, int head_dim,
                              void* stream) {
    dim3 threads(head_dim);
    dim3 blocks(seq_len, n_heads, batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    attention_output_kernel<<<blocks, threads, 0, s>>>(
        weights, V, output, batch, n_heads, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int32_t cuda_flash_attention(const float* Q, const float* K, const float* V, float* output,
                             int batch, int n_q_heads, int n_kv_heads, int seq_len,
                             int head_dim, float scale, void* stream) {
    constexpr int BLOCK_SIZE = 64;
    int smem_size = (2 * BLOCK_SIZE * head_dim + BLOCK_SIZE) * sizeof(float);
    dim3 threads(head_dim);
    dim3 blocks(seq_len, n_q_heads, batch);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    flash_attention_kernel<<<blocks, threads, smem_size, s>>>(
        Q, K, V, output, batch, n_q_heads, n_kv_heads, seq_len, head_dim, scale);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

} // extern "C"
