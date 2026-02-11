// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Multi-Query Attention with Rotary Position Embeddings (RoPE).
//!
//! Attention: scores = Q @ K^T / sqrt(d_k), weights = softmax(scores + mask), output = weights @ V
//!
//! Multi-Query Attention (MQA) / Grouped-Query Attention (GQA):
//! - n_heads query heads share n_kv_heads key/value heads
//! - Each query head h uses KV head (h % n_kv_heads), reducing KV cache by n_heads/n_kv_heads
//!
//! RoPE encodes position by rotating pairs of dimensions:
//!   [x0, x1] -> [x0*cos(theta) - x1*sin(theta), x0*sin(theta) + x1*cos(theta)]
//!   where theta_i = pos * base^(-2i/d)

use crate::config::Config;
use crate::layers::{Layer, Linear, collect_params, collect_params_mut};
use crate::tensor::{self, Shape, Tensor};

/// Multi-Query Attention with RoPE.
///
/// Projections: Q [hidden -> n_heads*head_dim], K/V [hidden -> n_kv_heads*head_dim]
/// Output: O [n_heads*head_dim -> hidden]
/// `scale` = 1/sqrt(head_dim) for dot-product scaling.
/// `freqs` = precomputed RoPE frequency bands: freq_i = 1 / base^(2i/head_dim)
pub struct MQAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    /// Precomputed RoPE frequencies: freq[i] = 1 / base^(2i/d)
    freqs: Vec<f32>,
    /// Cached forward data for backward
    last_input: Option<Vec<f32>>,
    last_input_shape: Option<Shape>,
    last_q: Option<Vec<f32>>,
    last_k: Option<Vec<f32>>,
    last_v: Option<Vec<f32>>,
    last_attn_weights: Option<Vec<f32>>,
    last_batch: usize,
    last_seq_len: usize,
    /// Pre-allocated backward scratch buffers (reused across steps)
    grad_w_row_buf: Option<Vec<f32>>,
    grad_q_buf: Option<Vec<f32>>,
    grad_k_buf: Option<Vec<f32>>,
    grad_v_buf: Option<Vec<f32>>,
    grad_scores_buf: Vec<f32>,
    inference_mode: bool,
    scores_buf: Vec<f32>,
}

impl MQAttention {
    pub fn new(config: &Config) -> Self {
        let hidden = config.hidden_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;

        // YaRN-style frequency scaling for extended context:
        // When alpha > 1, scale the base frequency to spread rotations over longer positions.
        // base' = base * alpha^(d / (d - 2))
        let base = if config.rope_alpha > 1.0 {
            config.rope_base
                * config
                    .rope_alpha
                    .powf(head_dim as f32 / (head_dim as f32 - 2.0))
        } else {
            config.rope_base
        };
        // Precompute frequency bands: freq_i = 1 / base^(2i/d) for i in [0, d/2)
        let half_dim = head_dim / 2;
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        Self {
            q_proj: Linear::new(hidden, n_heads * head_dim),
            k_proj: Linear::new(hidden, n_kv_heads * head_dim),
            v_proj: Linear::new(hidden, n_kv_heads * head_dim),
            o_proj: Linear::new(n_heads * head_dim, hidden),
            n_heads,
            n_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            freqs,
            last_input: None,
            last_input_shape: None,
            last_q: None,
            last_k: None,
            last_v: None,
            last_attn_weights: None,
            last_batch: 0,
            last_seq_len: 0,
            grad_w_row_buf: None,
            grad_q_buf: None,
            grad_k_buf: None,
            grad_v_buf: None,
            grad_scores_buf: Vec::new(),
            inference_mode: false,
            scores_buf: Vec::new(),
        }
    }

    /// Apply Rotary Position Embeddings (RoPE) to Q and K in-place.
    fn apply_rope(&self, q_data: &mut [f32], k_data: &mut [f32], batch: usize, seq_len: usize) {
        let half_dim = self.head_dim / 2;
        let rotate = |data: &mut [f32], heads: usize, b: usize, s: usize| {
            let base = (b * seq_len + s) * heads * self.head_dim;
            let pos = s as f32;
            for h in 0..heads {
                let off = base + h * self.head_dim;
                for i in 0..half_dim {
                    let angle = pos * self.freqs[i];
                    let (sin, cos) = angle.sin_cos();
                    let x0 = data[off + 2 * i];
                    let x1 = data[off + 2 * i + 1];
                    data[off + 2 * i] = x0 * cos - x1 * sin;
                    data[off + 2 * i + 1] = x0 * sin + x1 * cos;
                }
            }
        };

        for b in 0..batch {
            for s in 0..seq_len {
                rotate(q_data, self.n_heads, b, s);
                rotate(k_data, self.n_kv_heads, b, s);
            }
        }
    }
}

impl Layer for MQAttention {
    /// Full attention forward pass with caching for backward.
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let (batch, seq_len, _) = input.dims_3d();

        if !self.inference_mode {
            self.last_input = Some(input.data().to_vec());
            self.last_input_shape = Some(input.shape().clone());
        }
        self.last_batch = batch;
        self.last_seq_len = seq_len;

        // Step 1: Linear projections
        let mut q = self.q_proj.forward(input);
        let mut k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);

        // Step 2: Apply RoPE to Q and K (in-place to avoid allocation)
        self.apply_rope(q.data_mut(), k.data_mut(), batch, seq_len);

        if !self.inference_mode {
            self.last_q = Some(q.data().to_vec());
            self.last_k = Some(k.data().to_vec());
            self.last_v = Some(v.data().to_vec());
        }

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();
        let mut attn_out = vec![0.0; batch * seq_len * self.n_heads * self.head_dim];

        let scores_len = seq_len * seq_len;
        let mut scores = std::mem::take(&mut self.scores_buf);
        scores.resize(scores_len, 0.0);

        let mut all_weights = if self.inference_mode {
            Vec::new()
        } else {
            vec![0.0f32; batch * self.n_heads * seq_len * seq_len]
        };

        for b in 0..batch {
            for h in 0..self.n_heads {
                // GQA: query head h shares KV head (h % n_kv_heads)
                let kv_h = h % self.n_kv_heads;

                // Step 3-4: Compute scaled dot-product scores with causal masking
                for qi in 0..seq_len {
                    let q_off = ((b * seq_len + qi) * self.n_heads + h) * self.head_dim;

                    for ki in 0..=qi {
                        let k_off = ((b * seq_len + ki) * self.n_kv_heads + kv_h) * self.head_dim;
                        let mut dot = 0.0;
                        for d in 0..self.head_dim {
                            dot += q_data[q_off + d] * k_data[k_off + d];
                        }
                        scores[qi * seq_len + ki] = dot * self.scale;
                    }
                    for ki in (qi + 1)..seq_len {
                        scores[qi * seq_len + ki] = f32::NEG_INFINITY;
                    }
                }

                // Step 5: Softmax over the causal window.
                // Zero masked positions so backward BLAS sees 0, not -inf.
                for qi in 0..seq_len {
                    tensor::softmax_in_place(&mut scores[qi * seq_len..qi * seq_len + qi + 1]);
                    for ki in (qi + 1)..seq_len {
                        scores[qi * seq_len + ki] = 0.0;
                    }
                }

                if !self.inference_mode {
                    let w_off = (b * self.n_heads + h) * seq_len * seq_len;
                    all_weights[w_off..w_off + seq_len * seq_len].copy_from_slice(&scores);
                }

                // Step 6: Weighted sum of V
                for qi in 0..seq_len {
                    let out_off = ((b * seq_len + qi) * self.n_heads + h) * self.head_dim;
                    for ki in 0..=qi {
                        let w = scores[qi * seq_len + ki];
                        let v_off = ((b * seq_len + ki) * self.n_kv_heads + kv_h) * self.head_dim;
                        for d in 0..self.head_dim {
                            attn_out[out_off + d] += w * v_data[v_off + d];
                        }
                    }
                }
            }
        }

        self.scores_buf = scores;

        if !self.inference_mode {
            self.last_attn_weights = Some(all_weights);
        }

        // Step 7: Output projection
        let attn = Tensor::from_vec(
            attn_out,
            Shape::new(&[batch, seq_len, self.n_heads * self.head_dim]),
        );
        self.o_proj.forward(&attn)
    }

    /// Full attention backward pass:
    ///   1. Backward through W_o
    ///   2. grad_V = weights^T @ grad_attn_out per head
    ///   3. grad_weights = grad_attn_out @ V^T per head
    ///   4. Softmax backward
    ///   5. grad_Q = grad_scores @ K, grad_K = grad_scores^T @ Q per head
    ///   6. Reduce KV head gradients
    ///   7. Backward through W_q, W_k, W_v projections
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let batch = self.last_batch;
        let seq_len = self.last_seq_len;

        // 1. Backward through W_o (accumulates W_o weight grad)
        let grad_o_input = self.o_proj.backward(grad_output);

        // Restore shapes: grad_o_input is [batch, seq, n_heads*head_dim]
        let grad_o_data = grad_o_input.data();

        // Get cached data
        let q_data = self.last_q.take().unwrap_or_default();
        let k_data = self.last_k.take().unwrap_or_default();
        let v_data = self.last_v.take().unwrap_or_default();
        let attn_weights = self.last_attn_weights.take().unwrap_or_default();
        let input_data = self.last_input.take().unwrap_or_default();
        let input_shape = self
            .last_input_shape
            .take()
            .unwrap_or_else(|| Shape::new(&[]));

        // Reuse or allocate gradient buffers for Q, K, V
        // Layout: [batch, seq, heads, head_dim] flattened
        let q_total = batch * seq_len * self.n_heads * self.head_dim;
        let kv_total = batch * seq_len * self.n_kv_heads * self.head_dim;

        let mut grad_q = self.grad_q_buf.take().unwrap_or_default();
        grad_q.resize(q_total, 0.0);
        grad_q.fill(0.0);
        let mut grad_k = self.grad_k_buf.take().unwrap_or_default();
        grad_k.resize(kv_total, 0.0);
        grad_k.fill(0.0);
        let mut grad_v = self.grad_v_buf.take().unwrap_or_default();
        grad_v.resize(kv_total, 0.0);
        grad_v.fill(0.0);

        // Pre-allocate grad_scores scratch buffer [seq*seq]
        let mut grad_scores = std::mem::take(&mut self.grad_scores_buf);
        grad_scores.resize(seq_len * seq_len, 0.0);

        let hd = self.head_dim;
        let q_stride = self.n_heads * hd; // stride between rows in Q layout
        let kv_stride = self.n_kv_heads * hd; // stride between rows in K/V layout

        for b in 0..batch {
            for h in 0..self.n_heads {
                let kv_h = h % self.n_kv_heads;
                let w_off = (b * self.n_heads + h) * seq_len * seq_len;

                // Base offsets into strided arrays for this (b, h)
                let q_base = b * seq_len * q_stride + h * hd;
                let kv_base = b * seq_len * kv_stride + kv_h * hd;
                let go_base = q_base; // same layout as Q
                let gq_base = q_base;
                let gk_base = kv_base;
                let gv_base = kv_base;

                // 2. grad_V[b,kv_h] += W[b,h]^T @ dO[b,h]
                //    W: [S,S] contiguous at w_off, lda=S
                //    dO: [S,D] strided at go_base, ldb=q_stride
                //    gV: [S,D] strided at gv_base, ldc=kv_stride, beta=1.0 (GQA accumulate)
                crate::accelerate::sgemm_raw(
                    true,
                    false,
                    seq_len,
                    hd,
                    seq_len,
                    1.0,
                    &attn_weights[w_off..],
                    seq_len,
                    &grad_o_data[go_base..],
                    q_stride,
                    1.0,
                    &mut grad_v[gv_base..],
                    kv_stride,
                );

                // 3. grad_W = dO[b,h] @ V[b,kv_h]^T  → [S,S]
                crate::accelerate::sgemm_raw(
                    false,
                    true,
                    seq_len,
                    seq_len,
                    hd,
                    1.0,
                    &grad_o_data[go_base..],
                    q_stride,
                    &v_data[kv_base..],
                    kv_stride,
                    0.0,
                    &mut grad_scores,
                    seq_len,
                );

                // 4. Softmax backward (element-wise, stays scalar)
                // grad_scores = W * (grad_W - sum(grad_W * W, axis=-1))
                for qi in 0..seq_len {
                    let row = qi * seq_len;
                    let mut sum_gw_w = 0.0f32;
                    for ki in 0..=qi {
                        sum_gw_w += grad_scores[row + ki] * attn_weights[w_off + row + ki];
                    }
                    for ki in 0..=qi {
                        let w_val = attn_weights[w_off + row + ki];
                        grad_scores[row + ki] = w_val * (grad_scores[row + ki] - sum_gw_w);
                    }
                    // Zero future positions (sgemm wrote non-zero values there)
                    for ki in (qi + 1)..seq_len {
                        grad_scores[row + ki] = 0.0;
                    }
                }

                // 5. grad_Q[b,h] = scale * grad_scores @ K[b,kv_h]
                crate::accelerate::sgemm_raw(
                    false,
                    false,
                    seq_len,
                    hd,
                    seq_len,
                    self.scale,
                    &grad_scores,
                    seq_len,
                    &k_data[kv_base..],
                    kv_stride,
                    0.0,
                    &mut grad_q[gq_base..],
                    q_stride,
                );

                // 6. grad_K[b,kv_h] += scale * grad_scores^T @ Q[b,h]
                crate::accelerate::sgemm_raw(
                    true,
                    false,
                    seq_len,
                    hd,
                    seq_len,
                    self.scale,
                    &grad_scores,
                    seq_len,
                    &q_data[q_base..],
                    q_stride,
                    1.0,
                    &mut grad_k[gk_base..],
                    kv_stride,
                );
            }
        }

        // Stash scratch buffer for reuse
        self.grad_scores_buf = grad_scores;

        // 7. Backward through Q, K, V projections
        // from_vec consumes the buffers; we recover them afterward for reuse
        let grad_q_tensor = Tensor::from_vec(
            grad_q,
            Shape::new(&[batch, seq_len, self.n_heads * self.head_dim]),
        );
        let grad_k_tensor = Tensor::from_vec(
            grad_k,
            Shape::new(&[batch, seq_len, self.n_kv_heads * self.head_dim]),
        );
        let grad_v_tensor = Tensor::from_vec(
            grad_v,
            Shape::new(&[batch, seq_len, self.n_kv_heads * self.head_dim]),
        );

        // Set cached input for Q/K/V projections (they all used the same input x)
        // Share the same data: clone once for q, move original to k, recover from k for v
        self.q_proj.last_input = Some(input_data.clone());
        self.q_proj.last_batch = batch * seq_len;
        self.k_proj.last_input = Some(input_data);
        self.k_proj.last_batch = batch * seq_len;
        // v_proj reuses k_proj's data after k_proj backward (see below)
        self.v_proj.last_batch = batch * seq_len;

        let mut grad_x_q = self.q_proj.backward(&grad_q_tensor);
        let grad_x_k = self.k_proj.backward(&grad_k_tensor);
        // After k_proj backward, pass its cached input to v_proj (avoids a clone)
        self.v_proj.last_input = self.k_proj.last_input.take();
        let grad_x_v = self.v_proj.backward(&grad_v_tensor);

        // Recover buffers from consumed tensors for reuse on next step
        self.grad_q_buf = Some(grad_q_tensor.into_data());
        self.grad_k_buf = Some(grad_k_tensor.into_data());
        self.grad_v_buf = Some(grad_v_tensor.into_data());

        // Sum gradients from all projection paths (in-place to avoid allocations)
        grad_x_q.add_in_place(&grad_x_k);
        grad_x_q.add_in_place(&grad_x_v);
        grad_x_q
    }

    fn parameters(&self) -> Vec<&Tensor> {
        collect_params!(self.q_proj, self.k_proj, self.v_proj, self.o_proj)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        collect_params_mut!(self.q_proj, self.k_proj, self.v_proj, self.o_proj)
    }

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
        self.q_proj.set_inference_mode(mode);
        self.k_proj.set_inference_mode(mode);
        self.v_proj.set_inference_mode(mode);
        self.o_proj.set_inference_mode(mode);
    }
}
