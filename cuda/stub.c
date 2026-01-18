// Stub implementations when CUDA is not available
// All functions return error code -1 (CUDA not available)

#include <stdint.h>

#define STUB_IMPL(name, ...) \
    int32_t name(__VA_ARGS__) { return -1; }

// Elementwise
STUB_IMPL(cuda_silu, const float* input, float* output, int64_t n, void* stream)
STUB_IMPL(cuda_add, const float* a, const float* b, float* output, int64_t n, void* stream)
STUB_IMPL(cuda_mul, const float* a, const float* b, float* output, int64_t n, void* stream)
STUB_IMPL(cuda_scale, const float* input, float* output, float scale, int64_t n, void* stream)

// Softmax
STUB_IMPL(cuda_softmax, const float* input, float* output, int rows, int cols, void* stream)
STUB_IMPL(cuda_softmax_topk, const float* input, float* weights, int32_t* indices,
          int rows, int cols, int k, void* stream)

// RMSNorm
STUB_IMPL(cuda_rmsnorm, const float* input, const float* weight, float* output,
          int rows, int hidden_dim, float eps, void* stream)
STUB_IMPL(cuda_rmsnorm_residual, const float* input, const float* residual, const float* weight,
          float* output, float* residual_out, int rows, int hidden_dim, float eps, void* stream)

// GEMM
STUB_IMPL(cuda_gemm, const float* A, const float* B, float* C,
          int M, int N, int K, float alpha, float beta, void* stream)
STUB_IMPL(cuda_gemm_batched, const float* A, const float* B, float* C,
          int batch, int M, int N, int K, float alpha, float beta, void* stream)

// RoPE
STUB_IMPL(cuda_rope_freqs, float* freqs, int max_seq_len, int head_dim,
          float base, float alpha, void* stream)
STUB_IMPL(cuda_rope_forward, const float* input, const float* freqs, float* output,
          int batch, int seq_len, int n_heads, int head_dim, int offset, void* stream)
STUB_IMPL(cuda_rope_qk, const float* q, const float* k, const float* freqs,
          float* q_out, float* k_out, int batch, int seq_len, int n_q_heads, int n_kv_heads,
          int head_dim, int offset, void* stream)
STUB_IMPL(cuda_rope, float* q, float* k, const float* freqs,
          int batch, int seq_len, int n_heads, int head_dim, void* stream)
STUB_IMPL(cuda_rope_ntk, float* q, float* k, const float* freqs,
          int batch, int seq_len, int n_heads, int head_dim,
          float alpha, int orig_len, void* stream)

// Attention
STUB_IMPL(cuda_attention_scores, const float* Q, const float* K, float* scores,
          int batch, int n_heads, int seq_len, int head_dim, float scale, void* stream)
STUB_IMPL(cuda_attention_output, const float* weights, const float* V, float* output,
          int batch, int n_heads, int seq_len, int head_dim, void* stream)
STUB_IMPL(cuda_flash_attention, const float* Q, const float* K, const float* V, float* output,
          int batch, int seq_len, int n_heads, int head_dim, float scale, int is_causal, void* stream)
STUB_IMPL(cuda_mqa_attention, const float* Q, const float* K, const float* V, float* output,
          const float* mask, int batch, int seq_len, int n_heads, int head_dim, float scale, void* stream)

// Loss
STUB_IMPL(cuda_cross_entropy_forward, const float* logits, const int* targets,
          float* loss, float* log_probs, int batch_seq, int vocab_size, void* stream)
STUB_IMPL(cuda_cross_entropy_backward, const float* log_probs, const int* targets,
          float* d_logits, int batch_seq, int vocab_size, float scale, void* stream)
STUB_IMPL(cuda_aux_loss_forward, const float* router_probs, const int* expert_indices,
          float* aux_loss, float* f_counts, float* p_means,
          int batch_seq, int n_experts, int top_k, float alpha, void* stream)

// Optimizer
STUB_IMPL(cuda_adamw_step, float* param, const float* grad, float* m, float* v,
          float lr, float beta1, float beta2, float eps, float weight_decay,
          int step, int64_t size, void* stream)
STUB_IMPL(cuda_zero_grad, float* grad, int64_t size, void* stream)
STUB_IMPL(cuda_grad_clip, float* grad, float* partial_norms, float* total_norm,
          float clip_norm, int64_t size, void* stream)
STUB_IMPL(cuda_scatter_add, const float* grad_output, const int* indices,
          float* grad_weight, int num_indices, int embedding_dim, void* stream)

// Decode (GPU-side token generation)
STUB_IMPL(cuda_argmax, const float* logits, int32_t* output,
          int batch, int vocab_size, void* stream)
STUB_IMPL(cuda_sample, const float* logits, int32_t* output,
          const uint64_t* seeds, int batch, int vocab_size,
          float temperature, void* stream)
STUB_IMPL(cuda_topk_sample, const float* logits, int32_t* output,
          const uint64_t* seeds, int batch, int vocab_size,
          int k, float temperature, void* stream)
STUB_IMPL(cuda_topp_sample, const float* logits, int32_t* output,
          const uint64_t* seeds, int batch, int vocab_size,
          float top_p, float temperature, void* stream)
