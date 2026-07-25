// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

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

// Sgemm computes C = alpha*A@B + beta*C using Apple Accelerate cblas_sgemm.
// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major.
func Sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		C.float(alpha), (*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.float(beta), (*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}

// SgemmTransA computes C = alpha*A^T@B + beta*C using Apple Accelerate cblas_sgemm
// with CblasTrans on A.
func SgemmTransA(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
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

// SgemmTransB computes C = alpha*A@B^T + beta*C using Apple Accelerate cblas_sgemm
// with CblasTrans on B.
func SgemmTransB(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
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

// SgemmRaw is a direct cblas_sgemm wrapper with explicit trans flags and leading dimensions.
func SgemmRaw(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m == 0 || n == 0 || k == 0 {
		return
	}
	_ = a[:1]
	_ = b[:1]
	_ = c[:1]

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
