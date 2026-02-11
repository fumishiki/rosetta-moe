// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// CGO bridge to Apple Accelerate framework for hardware-accelerated BLAS.
//
// On Apple Silicon, cblas_sgemm routes through the AMX (Apple Matrix eXtensions)
// coprocessor, achieving 7-14x throughput vs NEON SIMD for matrix operations.
//
// CGO overhead: each call costs ~100ns-1us due to goroutine-to-C stack switching
// and Go runtime state saving. This is negligible for large matrices but can
// dominate for small sizes (e.g., 64x64 matmul ~524K FLOPS takes ~1us compute
// but ~1us CGO overhead = 50% overhead). Batch operations to amortize.

/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"
import "unsafe"

// sgemm computes C = alpha*A@B + beta*C using Apple Accelerate cblas_sgemm.
// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major.
//
// The early return on zero dimensions guards against empty-slice nil pointer
// dereference: (*C.float)(unsafe.Pointer(&a[0])) panics if a is empty.
// This is the most common CGO pitfall in Go -- always check len > 0 before
// taking &slice[0].
func sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	// unsafe.Pointer(&a[0]) extracts the raw data pointer from the Go slice header.
	// This pointer is pinned by CGO for the duration of the C call -- the GC will
	// not move the backing array while cblas_sgemm is executing.
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		C.float(alpha), (*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.float(beta), (*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}

// sgemmTransA computes C = alpha*A^T@B + beta*C using Apple Accelerate cblas_sgemm
// with CblasTrans on A. This avoids allocating a transposed copy of A.
// A: [k, m] row-major (stored as K rows of M cols, transposed to [m, k]), B: [k, n] row-major, C: [m, n] row-major.
//
// Used by Linear.Backward for dW = gradOutput^T @ input.
func sgemmTransA(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		C.float(alpha), (*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.float(beta), (*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}

// sgemmTransB computes C = alpha*A@B^T + beta*C using Apple Accelerate cblas_sgemm
// with CblasTrans on B. This avoids allocating a transposed copy of B.
// A: [m, k] row-major, B: [n, k] row-major (stored as N rows of K cols), C: [m, n] row-major.
//
// Used by Linear.Forward (weight stored as [out, in], need input @ weight^T).
func sgemmTransB(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
		C.int(m), C.int(n), C.int(k),
		C.float(alpha), (*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.float(beta), (*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}

// sgemmRaw is a direct cblas_sgemm wrapper with explicit trans flags and leading dimensions.
// Use for strided data views where the matrix is not contiguous in memory — e.g.,
// accessing a per-head [seq, headDim] slice from a [batch, seq, nHeads, headDim] array.
//
// transA, transB: whether to transpose A or B
// m, n, k: matrix dimensions (m x k) @ (k x n) = (m x n)
// alpha, beta: scaling factors for A@B and C
// lda, ldb, ldc: leading dimensions (stride in elements between rows)
//
// Example: For a [batch, seq, nHeads, headDim] tensor, to extract head h's [seq, headDim] matrix,
// the leading dimension is nHeads * headDim (the stride between rows in memory).
func sgemmRaw(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	// Pin Go slices so GC doesn't move them during FFI call.
	// The _ = slice[:1] pattern forces the compiler to emit a bounds check,
	// which keeps the backing array live for the duration of the C call.
	_ = a[:1]
	_ = b[:1]
	_ = c[:1]

	// Select transpose flags. We must pass C enum constants directly
	// because CGO generates them as specific types that can't be stored
	// in Go variables.
	if !transA && !transB {
		C.cblas_sgemm(
			C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
			C.int(m), C.int(n), C.int(k),
			C.float(alpha),
			(*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
			C.float(beta),
			(*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
		)
	} else if transA && !transB {
		C.cblas_sgemm(
			C.CblasRowMajor, C.CblasTrans, C.CblasNoTrans,
			C.int(m), C.int(n), C.int(k),
			C.float(alpha),
			(*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
			C.float(beta),
			(*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
		)
	} else if !transA && transB {
		C.cblas_sgemm(
			C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
			C.int(m), C.int(n), C.int(k),
			C.float(alpha),
			(*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
			C.float(beta),
			(*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
		)
	} else {
		C.cblas_sgemm(
			C.CblasRowMajor, C.CblasTrans, C.CblasTrans,
			C.int(m), C.int(n), C.int(k),
			C.float(alpha),
			(*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
			(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
			C.float(beta),
			(*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
		)
	}
}
