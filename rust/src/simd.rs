// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! NEON SIMD intrinsics for hot-path scalar operations.
//!
//! Provides approximate reciprocal square root (`vrsqrteq_f32` + Newton-Raphson)
//! matching Julia's `@fastmath` strategy that uses ARM NEON `frsqrte` for
//! approximate sqrt in AdamW and RMSNorm inner loops.
//!
//! Accuracy: ~23-bit mantissa after one Newton-Raphson refinement step,
//! sufficient for optimizer and normalization use cases.
//!
//! All unsafe is confined to NEON intrinsic calls with documented SAFETY invariants.
#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Approximate reciprocal square root for a slice: out[i] = 1/sqrt(v[i] + eps).
///
/// Uses NEON `vrsqrteq_f32` (hardware reciprocal sqrt estimate, ~12-bit)
/// followed by one Newton-Raphson refinement step for ~23-bit accuracy:
///   x1 = x0 * (3 - v * x0^2) / 2
///
/// Falls back to scalar `1.0 / (v + eps).sqrt()` for non-aarch64 or tail elements.
///
/// This is the same strategy Julia uses with `@fastmath` which lowers to
/// ARM NEON `frsqrte` + Newton-Raphson, giving ~4.58x speedup on AdamW.
#[cfg(target_arch = "aarch64")]
pub fn fast_rsqrt_slice(v: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(v.len(), out.len());
    let len = v.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: eps_vec is constructed from a valid f32 value.
    let eps_vec = unsafe { vdupq_n_f32(eps) };

    for i in 0..chunks {
        let base = i * 4;
        // SAFETY: base + 4 <= chunks * 4 <= len, so v[base..base+4] is valid.
        // vld1q_f32 reads 4 contiguous f32 values from an aligned-or-unaligned pointer.
        let val = unsafe { vld1q_f32(v.as_ptr().add(base)) };
        // val_eps = v + eps (prevent division by zero)
        let val_eps = unsafe { vaddq_f32(val, eps_vec) };
        // Initial estimate: ~12-bit accurate reciprocal sqrt
        let est = unsafe { vrsqrteq_f32(val_eps) };
        // Newton-Raphson step: est * vrsqrts(val_eps, est * est)
        // vrsqrtsq_f32 computes (3 - val_eps * est^2) / 2
        let est_sq = unsafe { vmulq_f32(est, est) };
        let refine = unsafe { vrsqrtsq_f32(val_eps, est_sq) };
        let result = unsafe { vmulq_f32(est, refine) };
        // SAFETY: base + 4 <= len, so out[base..base+4] is valid.
        unsafe { vst1q_f32(out.as_mut_ptr().add(base), result) };
    }

    // Scalar tail for remaining elements
    let tail_start = chunks * 4;
    for i in 0..remainder {
        out[tail_start + i] = 1.0 / (v[tail_start + i] + eps).sqrt();
    }
}

/// Non-aarch64 fallback: scalar reciprocal sqrt.
#[cfg(not(target_arch = "aarch64"))]
pub fn fast_rsqrt_slice(v: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(v.len(), out.len());
    for (o, &val) in out.iter_mut().zip(v.iter()) {
        *o = 1.0 / (val + eps).sqrt();
    }
}

/// In-place AdamW parameter update with SIMD approximate rsqrt.
///
/// For each element j:
///   m[j] = beta1 * m[j] + (1 - beta1) * g[j]
///   v[j] = beta2 * v[j] + (1 - beta2) * g[j]^2
///   m_hat = m[j] * corr1
///   v_hat = v[j] * corr2
///   p[j] -= lr * (m_hat * rsqrt(v_hat + eps) + wd * p[j])
///
/// The key optimization: replaces `m_hat / (v_hat.sqrt() + eps)` with
/// `m_hat * rsqrt(v_hat + eps)`, avoiding the expensive scalar sqrt.
/// rsqrt = 1/sqrt(x), computed via NEON hardware estimate + Newton-Raphson.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_simd(
    p: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    g: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    corr1: f32,
    corr2: f32,
) {
    let len = p.len();
    debug_assert_eq!(m.len(), len);
    debug_assert_eq!(v.len(), len);
    debug_assert_eq!(g.len(), len);

    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: All vdupq_n_f32 calls construct NEON vectors from valid f32 scalars.
    let beta1_vec = unsafe { vdupq_n_f32(beta1) };
    let beta2_vec = unsafe { vdupq_n_f32(beta2) };
    let one_minus_b1 = unsafe { vdupq_n_f32(1.0 - beta1) };
    let one_minus_b2 = unsafe { vdupq_n_f32(1.0 - beta2) };
    let corr1_vec = unsafe { vdupq_n_f32(corr1) };
    let corr2_vec = unsafe { vdupq_n_f32(corr2) };
    let eps_vec = unsafe { vdupq_n_f32(eps) };
    let lr_vec = unsafe { vdupq_n_f32(lr) };
    let wd_vec = unsafe { vdupq_n_f32(weight_decay) };

    for i in 0..chunks {
        let base = i * 4;
        // SAFETY: base + 4 <= chunks * 4 <= len. All loads/stores access valid memory
        // within the slices. NEON vld1q/vst1q handle unaligned access on aarch64.
        unsafe {
            let g_val = vld1q_f32(g.as_ptr().add(base));
            let m_val = vld1q_f32(m.as_ptr().add(base));
            let v_val = vld1q_f32(v.as_ptr().add(base));
            let p_val = vld1q_f32(p.as_ptr().add(base));

            // m = beta1 * m + (1 - beta1) * g
            let new_m = vmlaq_f32(vmulq_f32(beta1_vec, m_val), one_minus_b1, g_val);
            // v = beta2 * v + (1 - beta2) * g^2
            let g_sq = vmulq_f32(g_val, g_val);
            let new_v = vmlaq_f32(vmulq_f32(beta2_vec, v_val), one_minus_b2, g_sq);

            // Bias-corrected moments
            let m_hat = vmulq_f32(new_m, corr1_vec);
            let v_hat = vmulq_f32(new_v, corr2_vec);

            // Approximate rsqrt(v_hat + eps) via NEON estimate + Newton-Raphson
            let v_hat_eps = vaddq_f32(v_hat, eps_vec);
            let est = vrsqrteq_f32(v_hat_eps);
            let est_sq = vmulq_f32(est, est);
            let refine = vrsqrtsq_f32(v_hat_eps, est_sq);
            let rsqrt_val = vmulq_f32(est, refine);

            // update = lr * (m_hat * rsqrt(v_hat + eps) + wd * p)
            let adam_term = vmulq_f32(m_hat, rsqrt_val);
            let decay_term = vmulq_f32(wd_vec, p_val);
            let update = vmulq_f32(lr_vec, vaddq_f32(adam_term, decay_term));

            // p -= update
            let new_p = vsubq_f32(p_val, update);

            vst1q_f32(m.as_mut_ptr().add(base), new_m);
            vst1q_f32(v.as_mut_ptr().add(base), new_v);
            vst1q_f32(p.as_mut_ptr().add(base), new_p);
        }
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for j in 0..remainder {
        let idx = tail_start + j;
        let gj = g[idx];
        m[idx] = beta1 * m[idx] + (1.0 - beta1) * gj;
        v[idx] = beta2 * v[idx] + (1.0 - beta2) * gj * gj;
        let m_hat = m[idx] * corr1;
        let v_hat = v[idx] * corr2;
        p[idx] -= lr * (m_hat / (v_hat + eps).sqrt() + weight_decay * p[idx]);
    }
}

/// Non-aarch64 fallback: scalar AdamW step.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_simd(
    p: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    g: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    corr1: f32,
    corr2: f32,
) {
    for j in 0..p.len() {
        let gj = g[j];
        m[j] = beta1 * m[j] + (1.0 - beta1) * gj;
        v[j] = beta2 * v[j] + (1.0 - beta2) * gj * gj;
        let m_hat = m[j] * corr1;
        let v_hat = v[j] * corr2;
        p[j] -= lr * (m_hat / (v_hat + eps).sqrt() + weight_decay * p[j]);
    }
}
