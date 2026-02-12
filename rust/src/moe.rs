// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Mixture-of-Experts: Router, MoELayer, TransformerBlock.
//!
//! MoE: output = sum_k(gate_k * Expert_k(x)) for top-k experts
//!
//! Architecture:
//! - Router: learned linear gate that produces expert probabilities per token
//! - MoELayer: routes each token to top-k experts, weighted-sums their outputs
//! - TransformerBlock: Pre-norm residual block = Attention + MoE
//!
//! The auxiliary load-balancing loss encourages uniform expert utilization:
//!   L_aux = alpha * N * sum_e(f_e * p_e)
//! where f_e = fraction of tokens routed to expert e,
//!       p_e = mean gate probability for expert e.

use crate::attention::MQAttention;
use crate::config::Config;
use crate::layers::{ExpertFFN, Layer, Linear, RMSNorm, collect_params, collect_params_mut};
use crate::tensor::{Shape, Tensor};

/// Learned router that assigns tokens to experts.
///
/// gate: Linear [hidden_dim -> n_experts], followed by softmax to get probs.
/// Returns top-k expert indices and renormalized weights per token.
pub struct Router {
    gate: Linear,
    n_experts: usize,
    probs_buf: Vec<f32>,
    indices_buf: Vec<Vec<usize>>,
    selected_buf: Vec<bool>,
}

impl Router {
    pub fn new(hidden_dim: usize, n_experts: usize) -> Self {
        Self {
            gate: Linear::new(hidden_dim, n_experts),
            n_experts,
            probs_buf: Vec::new(),
            indices_buf: Vec::new(),
            selected_buf: vec![false; n_experts],
        }
    }

    /// Route each token to its top-k experts.
    ///
    /// Returns:
    ///   - indices: Vec<Vec<usize>> -- top-k expert indices per token
    ///   - weights: Tensor [batch_seq, top_k] -- renormalized gate weights
    ///   - raw_probs: Vec<f32> -- full softmax probs (for aux loss computation)
    ///
    /// Routing: probs = softmax(Gate(x)), select top-k, renormalize selected weights.
    /// Uses greedy top-k selection (O(K*E) per token, no sort allocation) matching Go.
    pub fn route(&mut self, input: &Tensor, top_k: usize) -> (Vec<Vec<usize>>, Tensor, Vec<f32>) {
        // Gate logits -> per-expert probabilities.
        // Reuse a flat probability buffer to avoid per-step tensor allocation.
        let gate_out = self.gate.forward(input);
        let (batch, seq_len, _) = input.dims_3d();
        let batch_seq = batch * seq_len;
        let logits_data = gate_out.data();
        let probs_len = batch_seq * self.n_experts;

        let mut probs = std::mem::take(&mut self.probs_buf);
        probs.resize(probs_len, 0.0);
        let probs_data = &mut probs[..probs_len];
        for t in 0..batch_seq {
            let start = t * self.n_experts;
            crate::tensor::softmax_into_slice(
                &logits_data[start..start + self.n_experts],
                &mut probs_data[start..start + self.n_experts],
            );
        }

        let mut all_indices = std::mem::take(&mut self.indices_buf);
        if all_indices.len() != batch_seq {
            all_indices.resize_with(batch_seq, Vec::new);
        }
        for idxs in &mut all_indices {
            if idxs.len() != top_k {
                idxs.resize(top_k, 0);
            }
        }
        let mut all_weights = vec![0.0; batch_seq * top_k];
        // Reusable selected flags -- avoids per-token allocation (matches Go pattern).
        let selected = &mut self.selected_buf;
        if selected.len() < self.n_experts {
            selected.resize(self.n_experts, false);
        }

        for t in 0..batch_seq {
            let row = &probs_data[t * self.n_experts..(t + 1) * self.n_experts];
            // Greedy top-k: pick highest, mark, repeat. O(K*E) per token, zero alloc.
            for s in selected.iter_mut() {
                *s = false;
            }
            for k in 0..top_k {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for e in 0..self.n_experts {
                    if !selected[e] && row[e] > best_val {
                        best_val = row[e];
                        best_idx = e;
                    }
                }
                selected[best_idx] = true;
                all_indices[t][k] = best_idx;
                all_weights[t * top_k + k] = best_val;
            }
            // Renormalize selected weights so they sum to 1
            let sum_w: f32 = all_weights[t * top_k..t * top_k + top_k]
                .iter()
                .sum::<f32>()
                .max(1e-12);
            let inv = 1.0 / sum_w;
            for k in 0..top_k {
                all_weights[t * top_k + k] *= inv;
            }
        }

        (
            all_indices,
            Tensor::from_vec(all_weights, Shape::new(&[batch_seq, top_k])),
            probs,
        )
    }
}

/// Cached routing data from the last forward pass, used for aux loss computation.
struct RouteData {
    gate_probs: Vec<f32>,
    indices: Vec<Vec<usize>>,
    batch_seq: usize,
    n_experts: usize,
    top_k: usize,
}

/// Mixture-of-Experts layer: routes tokens to top-k expert FFNs.
///
/// MoE(x) = sum_{k in top-k} weight_k * Expert_k(x)
///
/// `last_route` is stored in a RefCell to allow aux_loss() to be called on &self
/// after forward(). This is safe because forward() is single-threaded.
pub struct MoELayer {
    router: Router,
    experts: Vec<ExpertFFN>,
    top_k: usize,
    /// Cached routing data for aux loss.
    last_route: Option<RouteData>,
    /// Cached forward data for backward
    last_hidden: usize,
    last_batch_shape: Option<(usize, usize)>,
    /// Reused expert token groups from forward, consumed by backward.
    expert_tokens_buf: Vec<Vec<usize>>,
    expert_weight_idx_buf: Vec<Vec<usize>>,
    last_weights: Option<Vec<f32>>,
    inference_mode: bool,
}

impl MoELayer {
    pub fn new(config: &Config) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| ExpertFFN::new(config.hidden_dim, config.ffn_dim))
            .collect();
        let expert_tokens_buf = (0..config.n_experts).map(|_| Vec::new()).collect();
        let expert_weight_idx_buf = (0..config.n_experts).map(|_| Vec::new()).collect();
        Self {
            router: Router::new(config.hidden_dim, config.n_experts),
            experts,
            top_k: config.top_k_experts,
            last_route: None,
            last_hidden: 0,
            last_batch_shape: None,
            expert_tokens_buf,
            expert_weight_idx_buf,
            last_weights: None,
            inference_mode: false,
        }
    }

    /// Auxiliary load-balancing loss (Switch Transformer, Fedus et al. 2021).
    ///
    /// L_aux = alpha * N * sum_e(f_e * p_e)
    ///   f_e = (tokens assigned to expert e) / (total assignments)
    ///   p_e = mean(gate_prob_e) across all tokens
    ///   N   = number of experts
    ///
    /// Minimizing L_aux pushes f_e and p_e toward 1/N (uniform distribution),
    /// preventing expert collapse where a few experts get all the traffic.
    pub fn aux_loss(&self, alpha: f32) -> f32 {
        let data = match self.last_route.as_ref() {
            Some(d) => d,
            None => return 0.0,
        };

        let mut expert_counts = vec![0.0f32; data.n_experts];
        let mut expert_probs = vec![0.0f32; data.n_experts];

        for t in 0..data.batch_seq {
            for k in 0..data.top_k {
                expert_counts[data.indices[t][k]] += 1.0;
            }
            for e in 0..data.n_experts {
                expert_probs[e] += data.gate_probs[t * data.n_experts + e];
            }
        }

        // f_e = count_e / total_assignments, p_e = sum_probs_e / batch_seq
        let total_assign = (data.batch_seq * data.top_k) as f32;
        let denom_prob = data.batch_seq as f32;
        expert_counts
            .iter()
            .zip(&expert_probs)
            .map(|(c, p)| (c / total_assign) * (p / denom_prob))
            .sum::<f32>()
            * alpha
            * data.n_experts as f32
    }
}

impl Layer for MoELayer {
    /// MoE forward: batch tokens per expert, then weighted-sum.
    ///
    /// output[t] = sum_k(weight_k * Expert_k(input[t]))
    ///
    /// Batched dispatch (matching Julia/Go):
    ///   1. Router selects top-k experts per token with normalized weights
    ///   2. Build inverted index: expert -> list of assigned tokens
    ///   3. Gather tokens into per-expert contiguous batches
    ///   4. Run each expert once on its batch (single BLAS call vs N individual calls)
    ///   5. Scatter-add weighted expert outputs back to token positions
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let (batch, seq_len, hidden) = input.dims_3d();
        let batch_seq = batch * seq_len;

        let (indices, weights, gate_probs) = self.router.route(input, self.top_k);
        let w = weights.data();
        let input_data = input.data();
        let mut out = vec![0.0; batch_seq * hidden];

        // Build inverted index: expert_id -> list of (token_index, weight_slot)
        let n_experts = self.router.n_experts;
        if self.expert_tokens_buf.len() != n_experts {
            self.expert_tokens_buf.resize_with(n_experts, Vec::new);
        }
        if self.expert_weight_idx_buf.len() != n_experts {
            self.expert_weight_idx_buf.resize_with(n_experts, Vec::new);
        }
        for e in 0..n_experts {
            self.expert_tokens_buf[e].clear();
            self.expert_weight_idx_buf[e].clear();
        }

        for t in 0..batch_seq {
            for (k, &expert_idx) in indices[t].iter().enumerate() {
                self.expert_tokens_buf[expert_idx].push(t);
                self.expert_weight_idx_buf[expert_idx].push(k);
            }
        }

        // Process each expert's assigned tokens as a single batch
        for e_idx in 0..n_experts {
            let tokens = &self.expert_tokens_buf[e_idx];
            if tokens.is_empty() {
                continue;
            }
            let n_tok = tokens.len();

            // Gather: collect assigned token vectors into a contiguous batch
            let mut batch_data = vec![0.0f32; n_tok * hidden];
            for (i, &t) in tokens.iter().enumerate() {
                let src = &input_data[t * hidden..(t + 1) * hidden];
                batch_data[i * hidden..(i + 1) * hidden].copy_from_slice(src);
            }

            // Single batched expert forward (1 BLAS call instead of n_tok calls)
            let batch_input = Tensor::from_vec(batch_data, Shape::new(&[n_tok, 1, hidden]));
            let expert_out = self.experts[e_idx].forward(&batch_input);
            let e_out_data = expert_out.data();

            // Scatter-add: weighted expert output back to each token's position
            for (i, &t) in tokens.iter().enumerate() {
                let k = self.expert_weight_idx_buf[e_idx][i];
                let alpha = w[t * self.top_k + k];
                let t_off = t * hidden;
                let e_off = i * hidden;
                for d in 0..hidden {
                    out[t_off + d] += alpha * e_out_data[e_off + d];
                }
            }
        }

        if !self.inference_mode {
            self.last_hidden = hidden;
            self.last_batch_shape = Some((batch, seq_len));
            self.last_weights = Some(weights.into_data());
            self.last_route = Some(RouteData {
                gate_probs,
                indices,
                batch_seq,
                n_experts,
                top_k: self.top_k,
            });
        } else {
            self.router.probs_buf = gate_probs;
            self.router.indices_buf = indices;
            self.last_route = None;
            self.last_weights = None;
        }

        Tensor::from_vec(out, Shape::new(&[batch, seq_len, hidden]))
    }

    /// MoE backward: propagate gradients through each expert for the tokens it processed,
    /// weighted by the router weights. Accumulates expert parameter gradients.
    ///
    /// For each expert e with assigned tokens T_e:
    ///   expert_grad = weight_e * grad_output[t] for each t in T_e
    ///   grad_input[t] += expert.backward(expert_grad)[t]
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let (batch, seq_len) = self.last_batch_shape.unwrap_or((1, 1));
        let hidden = self.last_hidden;

        let flat_grad = grad_output.data();
        let route_data = self.last_route.take();
        let weights = self.last_weights.take().unwrap_or_default();
        if let Some(route) = route_data {
            self.router.probs_buf = route.gate_probs;
            self.router.indices_buf = route.indices;
        }

        let batch_seq = batch * seq_len;
        let mut grad_input = vec![0.0f32; batch_seq * hidden];
        let expert_tokens = &self.expert_tokens_buf;
        let expert_weight_idx = &self.expert_weight_idx_buf;

        for expert_idx in 0..self.router.n_experts {
            let token_indices = &expert_tokens[expert_idx];
            let weight_slots = &expert_weight_idx[expert_idx];
            if token_indices.is_empty() {
                continue;
            }
            let n_tok = token_indices.len();

            // Build weighted gradient batch for this expert
            let mut expert_grad_data = vec![0.0f32; n_tok * hidden];
            for (i, &t) in token_indices.iter().enumerate() {
                let k = weight_slots[i];
                let w = weights[t * self.top_k + k];
                let src_off = t * hidden;
                let dst_off = i * hidden;
                for d in 0..hidden {
                    expert_grad_data[dst_off + d] = flat_grad[src_off + d] * w;
                }
            }
            let expert_grad = Tensor::from_vec(expert_grad_data, Shape::new(&[n_tok, 1, hidden]));

            // Backward through expert (accumulates weight gradients)
            let grad_expert_input = self.experts[expert_idx].backward(&expert_grad);
            let ge_data = grad_expert_input.data();

            // Scatter-add input gradient back to token positions
            for (i, &t) in token_indices.iter().enumerate() {
                let src_off = i * hidden;
                let dst_off = t * hidden;
                for d in 0..hidden {
                    grad_input[dst_off + d] += ge_data[src_off + d];
                }
            }
        }

        Tensor::from_vec(grad_input, Shape::new(&[batch, seq_len, hidden]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.router.gate.weight];
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.router.gate.weight];
        for expert in &mut self.experts {
            params.extend(expert.parameters_mut());
        }
        params
    }

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
        self.router.gate.set_inference_mode(mode);
        for expert in &mut self.experts {
            expert.set_inference_mode(mode);
        }
    }
}

/// Pre-norm Transformer block with MoE FFN.
///
/// forward(x):
///   h = x + Attention(RMSNorm(x))     // residual + attention
///   y = h + MoE(RMSNorm(h))           // residual + MoE FFN
///
/// Pre-norm (norm before sublayer) is standard in modern LLMs for training stability.
pub struct TransformerBlock {
    attn_norm: RMSNorm,
    attention: MQAttention,
    ffn_norm: RMSNorm,
    pub(crate) moe: MoELayer,
    /// Cached intermediate h1 = x + attn(norm(x)) for backward residual flow
    last_h: Option<Vec<f32>>,
    last_h_shape: Option<Shape>,
    inference_mode: bool,
}

impl TransformerBlock {
    pub fn new(config: &Config) -> Self {
        Self {
            attn_norm: RMSNorm::new(config.hidden_dim),
            attention: MQAttention::new(config),
            ffn_norm: RMSNorm::new(config.hidden_dim),
            moe: MoELayer::new(config),
            last_h: None,
            last_h_shape: None,
            inference_mode: false,
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let normed1 = self.attn_norm.forward(input);
        let mut h = self.attention.forward(&normed1);
        h.add_in_place(input);

        if !self.inference_mode {
            self.last_h = Some(h.data().to_vec());
            self.last_h_shape = Some(h.shape().clone());
        }

        let normed2 = self.ffn_norm.forward(&h);
        let moe_out = self.moe.forward(&normed2);
        h.add_in_place(&moe_out);
        h
    }

    /// Backward through transformer block with proper residual gradient flow.
    ///
    /// Forward: h = x + Attention(RMSNorm_attn(x))
    ///          out = h + MoE(RMSNorm_ffn(h))
    ///
    /// Backward (reverse order):
    ///   grad_ffn_norm = ffn_norm.backward(grad_output)   -- through ffn_norm
    ///   grad_moe_input = moe.backward(grad_ffn_norm)     -- through MoE
    ///   grad_h = grad_output + grad_moe_input            -- residual connection
    ///
    ///   grad_attn_norm = attn_norm.backward(grad_h)      -- through attn_norm
    ///   grad_attn_input = attention.backward(grad_attn_norm) -- through attention
    ///   grad_x = grad_h + grad_attn_input                -- residual connection
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // MoE residual path: grad flows through ffn_norm -> moe, plus directly
        let grad_ffn_norm = self.ffn_norm.backward(grad_output);
        let grad_moe_input = self.moe.backward(&grad_ffn_norm);
        // Residual: grad_h = grad_output + grad through MoE path (in-place)
        let mut grad_h = grad_moe_input;
        grad_h.add_in_place(grad_output);

        // Attention residual path: grad flows through attn_norm -> attention, plus directly
        let grad_attn_norm = self.attn_norm.backward(&grad_h);
        let grad_attn_input = self.attention.backward(&grad_attn_norm);
        // Residual: grad_x = grad_h + grad through attention path (in-place)
        grad_h.add_in_place(&grad_attn_input);
        grad_h
    }

    fn parameters(&self) -> Vec<&Tensor> {
        collect_params!(self.attn_norm, self.attention, self.ffn_norm, self.moe)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        collect_params_mut!(self.attn_norm, self.attention, self.ffn_norm, self.moe)
    }

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
        self.attn_norm.set_inference_mode(mode);
        self.attention.set_inference_mode(mode);
        self.ffn_norm.set_inference_mode(mode);
        self.moe.set_inference_mode(mode);
    }
}

/// Standalone aux loss computation (same formula as MoELayer::aux_loss).
///
/// L_aux = alpha * N * sum_e(f_e * p_e)
/// Provided as a free function for use in testing and external callers.
pub fn compute_aux_loss(
    gate_probs: &[f32],
    indices: &[Vec<usize>],
    batch_seq: usize,
    n_experts: usize,
    top_k: usize,
    alpha: f32,
) -> f32 {
    let mut expert_counts = vec![0.0f32; n_experts];
    let mut expert_probs = vec![0.0f32; n_experts];

    for t in 0..batch_seq {
        for k in 0..top_k {
            expert_counts[indices[t][k]] += 1.0;
        }
        for e in 0..n_experts {
            expert_probs[e] += gate_probs[t * n_experts + e];
        }
    }

    let total_assign = (batch_seq * top_k) as f32;
    let denom_prob = batch_seq as f32;
    expert_counts
        .iter()
        .zip(&expert_probs)
        .map(|(c, p)| (c / total_assign) * (p / denom_prob))
        .sum::<f32>()
        * alpha
        * n_experts as f32
}
