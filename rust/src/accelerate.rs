// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Apple Accelerate BLAS bindings for sgemm.
//!
//! Links against the Accelerate.framework which uses the Apple AMX coprocessor
//! on Apple Silicon. This provides 7-14x speedup over naive NEON SIMD loops.
//!
//! All functions in this module are safe wrappers around `cblas_sgemm`, which
//! is the only unsafe FFI boundary. The unsafe block's correctness depends on:
//! 1. Slice bounds checked by debug_assert (callers guarantee sufficient length)
//! 2. Non-overlapping A, B, C regions (guaranteed by Rust's borrow rules)
//! 3. Leading dimensions (lda, ldb, ldc) matching row-major layout
//!
//! Row-major stride rules:
//! - No transpose: A is [M, K], lda = K (stride between rows)
//! - No transpose: B is [K, N], ldb = N
//! - B transposed: B is [N, K], ldb = K (stride of the un-transposed storage)
//! - Output C is always [M, N], ldc = N
#![allow(unsafe_code)]

// Link to Apple's Accelerate framework (provides BLAS via AMX coprocessor)
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// CBLAS enum constants
const ROW_MAJOR: i32 = 101; // CblasRowMajor
const NO_TRANS: i32 = 111; // CblasNoTrans
const TRANS: i32 = 112; // CblasTrans

/// Cast `usize` to `i32` with overflow guard.
///
/// BLAS uses i32 for dimensions. A silent truncation (e.g., usize 2^31 -> 0)
/// would cause cblas_sgemm to read/write wrong memory regions.
/// This is a hard assert (not debug_assert) because it's a safety boundary:
/// incorrect dimensions in BLAS = undefined behavior.
///
/// For this benchmark's matrices (max dim ~32768), this never fires.
#[inline(always)]
fn dim_i32(v: usize) -> i32 {
    assert!(
        v <= i32::MAX as usize,
        "matrix dimension {v} exceeds i32::MAX"
    );
    v as i32
}

/// C = alpha * A @ B + beta * C  (no transpose)
///
/// A: [m, k] row-major, B: [k, n] row-major, C: [m, n] row-major
/// Row-major leading dimensions: lda=k, ldb=n, ldc=n
/// (leading dim = number of columns = stride between consecutive rows)
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    debug_assert!(a.len() >= m * k, "a.len()={} < m*k={}", a.len(), m * k);
    debug_assert!(b.len() >= k * n, "b.len()={} < k*n={}", b.len(), k * n);
    debug_assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);

    let (mi, ni, ki) = (dim_i32(m), dim_i32(n), dim_i32(k));
    // SAFETY: Slice bounds are verified by debug_assert above.
    // a[..m*k], b[..k*n], c[..m*n] are valid, non-overlapping regions.
    // Row-major: lda=k, ldb=n, ldc=n.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            mi,
            ni,
            ki,
            alpha,
            a.as_ptr(),
            ki,
            b.as_ptr(),
            ni,
            beta,
            c.as_mut_ptr(),
            ni,
        );
    }
}

/// C = alpha * A^T @ B + beta * C  (A transposed)
///
/// A: [k, m] row-major (stored transposed), B: [k, n] row-major, C: [m, n] row-major
/// Key stride difference: lda=m (not k) because A is stored as [k, m] and
/// CBLAS reads it as the transpose [m, k]. This is the Linear backward path where
/// we compute dW = grad_output^T @ input.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_transa(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    debug_assert!(a.len() >= k * m, "a.len()={} < k*m={}", a.len(), k * m);
    debug_assert!(b.len() >= k * n, "b.len()={} < k*n={}", b.len(), k * n);
    debug_assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);

    let (mi, ni, ki) = (dim_i32(m), dim_i32(n), dim_i32(k));
    // SAFETY: Slice bounds are verified by debug_assert above.
    // a[..k*m], b[..k*n], c[..m*n] are valid, non-overlapping regions.
    // Row-major with A transposed: lda=m, ldb=n, ldc=n.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            TRANS,
            NO_TRANS,
            mi,
            ni,
            ki,
            alpha,
            a.as_ptr(),
            mi,
            b.as_ptr(),
            ni,
            beta,
            c.as_mut_ptr(),
            ni,
        );
    }
}

/// C = alpha * A @ B^T + beta * C  (B transposed)
///
/// A: [m, k] row-major, B: [n, k] row-major (stored transposed), C: [m, n] row-major
/// Key stride difference: ldb=k (not n) because B is stored as [n, k] and
/// CBLAS reads it as the transpose. This is the Linear forward path where
/// weight W is [out_features, in_features] and we compute input @ W^T.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_transb(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    debug_assert!(a.len() >= m * k, "a.len()={} < m*k={}", a.len(), m * k);
    debug_assert!(b.len() >= n * k, "b.len()={} < n*k={}", b.len(), n * k);
    debug_assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);

    let (mi, ni, ki) = (dim_i32(m), dim_i32(n), dim_i32(k));
    // SAFETY: Slice bounds are verified by debug_assert above.
    // a[..m*k], b[..n*k], c[..m*n] are valid, non-overlapping regions.
    // Row-major with B transposed: lda=k, ldb=k, ldc=n.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            mi,
            ni,
            ki,
            alpha,
            a.as_ptr(),
            ki,
            b.as_ptr(),
            ki,
            beta,
            c.as_mut_ptr(),
            ni,
        );
    }
}

/// Direct cblas_sgemm wrapper with explicit trans flags and leading dimensions.
///
/// Unlike the convenience wrappers (sgemm, sgemm_transa, sgemm_transb) which
/// compute lda/ldb/ldc from dimensions, this function accepts them directly.
/// Use this for strided views where the data is not contiguous — e.g., accessing
/// a per-head [seq, head_dim] slice from a [batch, seq, n_heads, head_dim] array.
///
/// Row-major leading dimension rules:
///   - No trans A[M,K]: lda >= K, element A[i,j] at offset i*lda + j
///   - Trans A (stored [K,M]): lda >= M, element at j*lda + i
///   - Output C[M,N]: ldc >= N
#[allow(clippy::too_many_arguments)]
pub fn sgemm_raw(
    trans_a: bool,
    trans_b: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    let ta = if trans_a { TRANS } else { NO_TRANS };
    let tb = if trans_b { TRANS } else { NO_TRANS };
    let (mi, ni, ki) = (dim_i32(m), dim_i32(n), dim_i32(k));
    // SAFETY: Caller guarantees valid slice ranges via strided layout.
    // For strided data, the slice extends beyond the logical matrix
    // (e.g., &data[offset..] for a sub-view). BLAS only accesses
    // elements within the M×lda (or K×lda for trans) stride pattern.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            ta,
            tb,
            mi,
            ni,
            ki,
            alpha,
            a.as_ptr(),
            dim_i32(lda),
            b.as_ptr(),
            dim_i32(ldb),
            beta,
            c.as_mut_ptr(),
            dim_i32(ldc),
        );
    }
}
