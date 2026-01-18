// Package cuda provides Go bindings to CUDA kernels via cgo.
//
// Build the C library first:
//
//	cd go/cuda && make
//
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/../../cuda/kernels
#cgo LDFLAGS: -L${SRCDIR}/lib -lcudann

#include <stdint.h>

// Forward declarations (matches stub.c / CUDA kernels)
extern int32_t cuda_silu(const float* input, float* output, int64_t n, void* stream);
extern int32_t cuda_add(const float* a, const float* b, float* output, int64_t n, void* stream);
extern int32_t cuda_mul(const float* a, const float* b, float* output, int64_t n, void* stream);
extern int32_t cuda_scale(const float* input, float* output, float scale, int64_t n, void* stream);

extern int32_t cuda_softmax(const float* input, float* output, int batch, int dim, void* stream);
extern int32_t cuda_softmax_topk(const float* input, float* values, int32_t* indices, int batch, int dim, int k, void* stream);

extern int32_t cuda_rmsnorm(const float* input, const float* weight, float* output, int batch, int dim, float eps, void* stream);
extern int32_t cuda_rmsnorm_residual(const float* input, const float* weight, const float* residual, float* output, int batch, int dim, float eps, void* stream);

extern int32_t cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta, void* stream);
extern int32_t cuda_gemm_batched(const float* A, const float* B, float* C, int batch, int M, int N, int K, float alpha, float beta, void* stream);

extern int32_t cuda_rope(float* q, float* k, const float* freqs, int batch, int seq_len, int n_heads, int head_dim, void* stream);
extern int32_t cuda_rope_ntk(float* q, float* k, const float* freqs, int batch, int seq_len, int n_heads, int head_dim, float alpha, int orig_len, void* stream);

extern int32_t cuda_mqa_attention(const float* Q, const float* K, const float* V, float* output, const float* mask, int batch, int seq_len, int n_heads, int head_dim, float scale, void* stream);
extern int32_t cuda_flash_attention(const float* Q, const float* K, const float* V, float* output, int batch, int seq_len, int n_heads, int head_dim, float scale, int is_causal, void* stream);

extern int32_t cuda_cross_entropy_forward(const float* logits, const int32_t* targets, float* loss, float* log_probs, int batch, int vocab_size, void* stream);
extern int32_t cuda_cross_entropy_backward(const float* log_probs, const int32_t* targets, float* grad, int batch, int vocab_size, void* stream);
extern int32_t cuda_aux_loss_forward(const float* router_probs, float* loss, int batch, int n_experts, int top_k, float alpha, void* stream);

extern int32_t cuda_adamw_step(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int step, int64_t size, void* stream);
extern int32_t cuda_zero_grad(float* grad, int64_t size, void* stream);
extern int32_t cuda_grad_clip(float* grad, float* partial_norms, float* total_norm, float clip_norm, int64_t size, void* stream);
extern int32_t cuda_scatter_add(const float* grad_output, const int32_t* indices, float* grad_weight, int num_indices, int embedding_dim, void* stream);

extern int32_t cuda_argmax(const float* logits, int32_t* output, int batch, int vocab_size, void* stream);
extern int32_t cuda_sample(const float* logits, int32_t* output, const uint64_t* seeds, int batch, int vocab_size, float temperature, void* stream);
extern int32_t cuda_topk_sample(const float* logits, int32_t* output, const uint64_t* seeds, int batch, int vocab_size, int k, float temperature, void* stream);
extern int32_t cuda_topp_sample(const float* logits, int32_t* output, const uint64_t* seeds, int batch, int vocab_size, float top_p, float temperature, void* stream);
*/
import "C"
import (
	"errors"
	"unsafe"
)

// CudaError represents a CUDA error.
var ErrCudaNotAvailable = errors.New("CUDA not available")

// Stream represents a CUDA stream (nil for default stream).
type Stream unsafe.Pointer

// DefaultStream is the default CUDA stream.
var DefaultStream Stream = nil

// checkResult converts a CUDA return code to an error.
func checkResult(code C.int32_t) error {
	if code != 0 {
		return ErrCudaNotAvailable
	}
	return nil
}

// SiLU applies SiLU activation on GPU.
func SiLU(input, output []float32, stream Stream) error {
	if len(input) != len(output) {
		return errors.New("input and output must have same length")
	}
	return checkResult(C.cuda_silu(
		(*C.float)(&input[0]),
		(*C.float)(&output[0]),
		C.int64_t(len(input)),
		unsafe.Pointer(stream),
	))
}

// Add performs element-wise addition on GPU.
func Add(a, b, output []float32, stream Stream) error {
	if len(a) != len(b) || len(a) != len(output) {
		return errors.New("all arrays must have same length")
	}
	return checkResult(C.cuda_add(
		(*C.float)(&a[0]),
		(*C.float)(&b[0]),
		(*C.float)(&output[0]),
		C.int64_t(len(a)),
		unsafe.Pointer(stream),
	))
}

// Mul performs element-wise multiplication on GPU.
func Mul(a, b, output []float32, stream Stream) error {
	if len(a) != len(b) || len(a) != len(output) {
		return errors.New("all arrays must have same length")
	}
	return checkResult(C.cuda_mul(
		(*C.float)(&a[0]),
		(*C.float)(&b[0]),
		(*C.float)(&output[0]),
		C.int64_t(len(a)),
		unsafe.Pointer(stream),
	))
}

// Scale multiplies by scalar on GPU.
func Scale(input, output []float32, scale float32, stream Stream) error {
	if len(input) != len(output) {
		return errors.New("input and output must have same length")
	}
	return checkResult(C.cuda_scale(
		(*C.float)(&input[0]),
		(*C.float)(&output[0]),
		C.float(scale),
		C.int64_t(len(input)),
		unsafe.Pointer(stream),
	))
}

// Softmax applies softmax on GPU.
func Softmax(input, output []float32, batch, dim int, stream Stream) error {
	return checkResult(C.cuda_softmax(
		(*C.float)(&input[0]),
		(*C.float)(&output[0]),
		C.int(batch),
		C.int(dim),
		unsafe.Pointer(stream),
	))
}

// RMSNorm applies RMS normalization on GPU.
func RMSNorm(input, weight, output []float32, batch, dim int, eps float32, stream Stream) error {
	return checkResult(C.cuda_rmsnorm(
		(*C.float)(&input[0]),
		(*C.float)(&weight[0]),
		(*C.float)(&output[0]),
		C.int(batch),
		C.int(dim),
		C.float(eps),
		unsafe.Pointer(stream),
	))
}

// GEMM performs matrix multiplication on GPU: C = alpha * A @ B + beta * C.
func GEMM(A, B, C []float32, M, N, K int, alpha, beta float32, stream Stream) error {
	return checkResult(C.cuda_gemm(
		(*C.float)(&A[0]),
		(*C.float)(&B[0]),
		(*C.float)(&C[0]),
		C.int(M),
		C.int(N),
		C.int(K),
		C.float(alpha),
		C.float(beta),
		unsafe.Pointer(stream),
	))
}

// GEMMBatched performs batched matrix multiplication on GPU.
func GEMMBatched(A, B, C []float32, batch, M, N, K int, alpha, beta float32, stream Stream) error {
	return checkResult(C.cuda_gemm_batched(
		(*C.float)(&A[0]),
		(*C.float)(&B[0]),
		(*C.float)(&C[0]),
		C.int(batch),
		C.int(M),
		C.int(N),
		C.int(K),
		C.float(alpha),
		C.float(beta),
		unsafe.Pointer(stream),
	))
}

// CrossEntropyForward computes cross entropy loss on GPU.
func CrossEntropyForward(logits []float32, targets []int32, loss, logProbs []float32, batch, vocabSize int, stream Stream) error {
	return checkResult(C.cuda_cross_entropy_forward(
		(*C.float)(&logits[0]),
		(*C.int32_t)(&targets[0]),
		(*C.float)(&loss[0]),
		(*C.float)(&logProbs[0]),
		C.int(batch),
		C.int(vocabSize),
		unsafe.Pointer(stream),
	))
}

// AdamWStep performs AdamW optimizer step on GPU.
func AdamWStep(param, grad, m, v []float32, lr, beta1, beta2, eps, weightDecay float32, step int, stream Stream) error {
	return checkResult(C.cuda_adamw_step(
		(*C.float)(&param[0]),
		(*C.float)(&grad[0]),
		(*C.float)(&m[0]),
		(*C.float)(&v[0]),
		C.float(lr),
		C.float(beta1),
		C.float(beta2),
		C.float(eps),
		C.float(weightDecay),
		C.int(step),
		C.int64_t(len(param)),
		unsafe.Pointer(stream),
	))
}

// Argmax performs greedy decoding on GPU.
func Argmax(logits []float32, output []int32, batch, vocabSize int, stream Stream) error {
	return checkResult(C.cuda_argmax(
		(*C.float)(&logits[0]),
		(*C.int32_t)(&output[0]),
		C.int(batch),
		C.int(vocabSize),
		unsafe.Pointer(stream),
	))
}

// Sample performs multinomial sampling on GPU.
func Sample(logits []float32, output []int32, seeds []uint64, batch, vocabSize int, temperature float32, stream Stream) error {
	return checkResult(C.cuda_sample(
		(*C.float)(&logits[0]),
		(*C.int32_t)(&output[0]),
		(*C.uint64_t)(&seeds[0]),
		C.int(batch),
		C.int(vocabSize),
		C.float(temperature),
		unsafe.Pointer(stream),
	))
}

// TopKSample performs top-k sampling on GPU.
func TopKSample(logits []float32, output []int32, seeds []uint64, batch, vocabSize, k int, temperature float32, stream Stream) error {
	return checkResult(C.cuda_topk_sample(
		(*C.float)(&logits[0]),
		(*C.int32_t)(&output[0]),
		(*C.uint64_t)(&seeds[0]),
		C.int(batch),
		C.int(vocabSize),
		C.int(k),
		C.float(temperature),
		unsafe.Pointer(stream),
	))
}

// TopPSample performs nucleus (top-p) sampling on GPU.
func TopPSample(logits []float32, output []int32, seeds []uint64, batch, vocabSize int, topP, temperature float32, stream Stream) error {
	return checkResult(C.cuda_topp_sample(
		(*C.float)(&logits[0]),
		(*C.int32_t)(&output[0]),
		(*C.uint64_t)(&seeds[0]),
		C.int(batch),
		C.int(vocabSize),
		C.float(topP),
		C.float(temperature),
		unsafe.Pointer(stream),
	))
}
