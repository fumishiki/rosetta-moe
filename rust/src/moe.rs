// SPDX-License-Identifier: CC-BY-4.0
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

use std::cell::RefCell;

use crate::attention::MQAttention;
use crate::config::Config;
use crate::layers::{collect_params, collect_params_mut, ExpertFFN, Layer, Linear, RMSNorm};
use crate::tensor::{Shape, Tensor};

/// Learned router that assigns tokens to experts.
///
/// gate: Linear [hidden_dim -> n_experts], followed by softmax to get probs.
/// Returns top-k expert indices and renormalized weights per token.
pub struct Router {
    gate: Linear,
    n_experts: usize,
}

impl Router {
    pub fn new(hidden_dim: usize, n_experts: usize) -> Self {
        Self {
            gate: Linear::new(hidden_dim, n_experts),
            n_experts,
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
        // Gate logits -> softmax -> per-expert probabilities
        // Use softmax_into to avoid allocating a new tensor
        let gate_out = self.gate.forward(input);
        let mut probs = Tensor::zeros(gate_out.shape().clone(), gate_out.dtype());
        if let Err(e) = gate_out.softmax_into(&mut probs) {
            panic!("{e}");
        }
        let (batch, seq_len, _) = input.dims_3d();
        let batch_seq = batch * seq_len;
        let probs_data = probs.data();

        let mut all_indices = Vec::with_capacity(batch_seq);
        let mut all_weights = vec![0.0; batch_seq * top_k];
        // Reusable selected flags -- avoids per-token allocation (matches Go pattern)
        let mut selected = vec![false; self.n_experts];

        for t in 0..batch_seq {
            let row = &probs_data[t * self.n_experts..(t + 1) * self.n_experts];
            // Greedy top-k: pick highest, mark, repeat. O(K*E) per token, zero alloc.
            for s in selected.iter_mut() {
                *s = false;
            }
            let mut idxs = Vec::with_capacity(top_k);
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
                idxs.push(best_idx);
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
            all_indices.push(idxs);
        }

        (
            all_indices,
            Tensor::from_vec(all_weights, Shape::new(&[batch_seq, top_k])),
            probs_data.to_vec(),
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
    // RefCell: forward() needs &self (Layer trait) but must store route data.
    // Single-threaded access makes this safe.
    last_route: RefCell<Option<RouteData>>,
    /// Cached forward data for backward
    last_input: Option<Vec<f32>>,
    last_hidden: usize,
    last_batch_shape: Option<(usize, usize)>,
    last_indices: Option<Vec<Vec<usize>>>,
    last_weights: Option<Vec<f32>>,
    /// Pre-allocated backward scratch buffers (reused across steps)
    token_indices_buf: Vec<usize>,
    weight_slots_buf: Vec<usize>,
    inference_mode: bool,
}

impl MoELayer {
    pub fn new(config: &Config) -> Self {
        let experts = (0..config.n_experts)
            .map(|_| ExpertFFN::new(config.hidden_dim, config.ffn_dim))
            .collect();
        Self {
            router: Router::new(config.hidden_dim, config.n_experts),
            experts,
            top_k: config.top_k_experts,
            last_route: RefCell::new(None),
            last_input: None,
            last_hidden: 0,
            last_batch_shape: None,
            last_indices: None,
            last_weights: None,
            token_indices_buf: Vec::new(),
            weight_slots_buf: Vec::new(),
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
        let guard = self.last_route.borrow();
        let data = match guard.as_ref() {
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
        let mut expert_tokens: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
        let mut expert_weight_idx: Vec<Vec<usize>> = vec![Vec::new(); n_experts];

        for t in 0..batch_seq {
            for (k, &expert_idx) in indices[t].iter().enumerate() {
                expert_tokens[expert_idx].push(t);
                expert_weight_idx[expert_idx].push(k);
            }
        }

        // Process each expert's assigned tokens as a single batch
        for e_idx in 0..n_experts {
            let tokens = &expert_tokens[e_idx];
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
                let k = expert_weight_idx[e_idx][i];
                let alpha = w[t * self.top_k + k];
                let t_off = t * hidden;
                let e_off = i * hidden;
                for d in 0..hidden {
                    out[t_off + d] += alpha * e_out_data[e_off + d];
                }
            }
        }

        if !self.inference_mode {
            self.last_input = Some(input_data.to_vec());
            self.last_hidden = hidden;
            self.last_batch_shape = Some((batch, seq_len));
            self.last_indices = Some(indices.clone());
            self.last_weights = Some(w.to_vec());

            *self.last_route.borrow_mut() = Some(RouteData {
                gate_probs,
                indices,
                batch_seq,
                n_experts,
                top_k: self.top_k,
            });
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
        let batch_seq = batch * seq_len;

        let flat_grad = grad_output.data();
        let indices = self.last_indices.take().unwrap_or_default();
        let weights = self.last_weights.take().unwrap_or_default();
        let flat_x = self.last_input.take().unwrap_or_default();

        let mut grad_input = vec![0.0f32; batch_seq * hidden];

        let n_experts = self.router.n_experts;

        // Take buffers out of self to avoid borrow conflicts
        let mut token_indices = std::mem::take(&mut self.token_indices_buf);
        let mut weight_slots = std::mem::take(&mut self.weight_slots_buf);

        for expert_idx in 0..n_experts {
            // Find tokens assigned to this expert and their weight slots (reuse buffers)
            token_indices.clear();
            weight_slots.clear();
            for t in 0..batch_seq {
                for (k, &e) in indices[t].iter().enumerate() {
                    if e == expert_idx {
                        token_indices.push(t);
                        weight_slots.push(k);
                    }
                }
            }
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

            // Restore expert's cached input (expert forward saw gathered tokens)
            let mut expert_input_data = vec![0.0f32; n_tok * hidden];
            for (i, &t) in token_indices.iter().enumerate() {
                let src = &flat_x[t * hidden..(t + 1) * hidden];
                expert_input_data[i * hidden..(i + 1) * hidden].copy_from_slice(src);
            }

            // Restore cached input for the expert's sub-layers
            // Only store once on expert; SwiGLU backward handles sub-layer forwarding
            let expert = &mut self.experts[expert_idx];
            expert.last_input = Some(expert_input_data);
            expert.last_input_shape = Some(Shape::new(&[n_tok, 1, hidden]));
            // gate_proj and up_proj last_input will be set by SwiGLU backward
            // from expert.last_input (avoids 2 clones)
            expert.gate_proj.last_batch = n_tok;
            expert.up_proj.last_batch = n_tok;

            // Backward through expert (accumulates weight gradients)
            let grad_expert_input = expert.backward(&expert_grad);
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

        // Stash reusable buffers back
        self.token_indices_buf = token_indices;
        self.weight_slots_buf = weight_slots;

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
