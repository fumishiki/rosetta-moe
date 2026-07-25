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

use super::attention::MQAttention;
use crate::config::Config;
use super::layers::{ExpertFFN, Layer, Linear, RMSNorm, collect_params, collect_params_mut};
use super::tensor::{Shape, Tensor};
use super::train::RoutingMode;

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
    /// Cached pre-softmax logits for z-loss computation
    last_logits: Vec<f32>,
    /// Cached gate probabilities (post-softmax) for z-loss gradient
    last_gate_probs: Vec<f32>,
    routing_mode: RoutingMode,
    expert_bias: Vec<f32>,
    relu_lambda_l1: f32,
    last_avg_active: f32,
    last_expert_counts: Vec<f32>,
    last_relu_sum: f32,
    /// Reusable buffer for router weights (avoids alloc per forward)
    weights_buf: Vec<f32>,
    /// Reusable buffer for BiasFree selection scores (avoids per-token alloc)
    selection_buf: Vec<f32>,
    /// Reusable buffer for ReLU active expert list (avoids per-token alloc)
    active_buf: Vec<(usize, f32)>,
}

impl Router {
    pub fn new(hidden_dim: usize, n_experts: usize) -> Self {
        Self {
            gate: Linear::new(hidden_dim, n_experts),
            n_experts,
            probs_buf: Vec::new(),
            indices_buf: Vec::new(),
            selected_buf: vec![false; n_experts],
            last_logits: Vec::new(),
            last_gate_probs: Vec::new(),
            routing_mode: RoutingMode::default(),
            expert_bias: vec![0.0; n_experts],
            relu_lambda_l1: 0.01,
            last_avg_active: 0.0,
            last_expert_counts: vec![0.0; n_experts],
            last_relu_sum: 0.0,
            weights_buf: Vec::new(),
            selection_buf: vec![0.0; n_experts],
            active_buf: Vec::with_capacity(n_experts),
        }
    }

    pub(crate) fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Read-only access to the gate weight data (for GPU weight upload).
    pub(crate) fn gate_weight_data(&self) -> &[f32] {
        self.gate.weight.data()
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
        let gate_out = self.gate.forward(input);
        let (batch, seq_len, _) = input.dims_3d();
        let batch_seq = batch * seq_len;
        let logits_data = gate_out.data();

        match self.routing_mode {
            RoutingMode::TopK | RoutingMode::BiasFree => {
                self.route_topk_or_biasfree(batch_seq, logits_data, top_k)
            }
            RoutingMode::ReLU => self.route_relu(batch_seq, logits_data, top_k),
        }
    }

    fn route_topk_or_biasfree(
        &mut self,
        batch_seq: usize,
        logits_data: &[f32],
        top_k: usize,
    ) -> (Vec<Vec<usize>>, Tensor, Vec<f32>) {
        let probs_len = batch_seq * self.n_experts;
        self.last_logits.clear();
        self.last_logits.extend_from_slice(logits_data);

        let mut probs = std::mem::take(&mut self.probs_buf);
        probs.resize(probs_len, 0.0);
        let probs_data = &mut probs[..probs_len];
        for t in 0..batch_seq {
            let start = t * self.n_experts;
            super::tensor::softmax_into_slice(
                &logits_data[start..start + self.n_experts],
                &mut probs_data[start..start + self.n_experts],
            );
        }

        self.last_gate_probs.clear();
        self.last_gate_probs.extend_from_slice(probs_data);

        let mut all_indices = std::mem::take(&mut self.indices_buf);
        if all_indices.len() != batch_seq {
            all_indices.resize_with(batch_seq, Vec::new);
        }
        for idxs in &mut all_indices {
            if idxs.len() != top_k {
                idxs.resize(top_k, 0);
            }
        }
        let mut all_weights = std::mem::take(&mut self.weights_buf);
        all_weights.resize(batch_seq * top_k, 0.0);
        all_weights.fill(0.0);
        let selected = &mut self.selected_buf;
        if selected.len() < self.n_experts {
            selected.resize(self.n_experts, false);
        }

        // Clear expert counts for this routing pass
        self.last_expert_counts.clear();
        self.last_expert_counts.resize(self.n_experts, 0.0);

        let mut selection_scores = std::mem::take(&mut self.selection_buf);
        selection_scores.resize(self.n_experts, 0.0);

        for t in 0..batch_seq {
            let row = &probs_data[t * self.n_experts..(t + 1) * self.n_experts];

            // BiasFree: augment selection scores with expert bias
            if matches!(self.routing_mode, RoutingMode::BiasFree) {
                for (i, (p, b)) in row.iter().zip(&self.expert_bias).enumerate() {
                    selection_scores[i] = p + b;
                }
            } else {
                selection_scores[..self.n_experts].copy_from_slice(row);
            }

            for s in selected.iter_mut() {
                *s = false;
            }
            for k in 0..top_k {
                let mut best_idx = 0;
                let mut best_val = f32::NEG_INFINITY;
                for e in 0..self.n_experts {
                    if !selected[e] && selection_scores[e] > best_val {
                        best_val = selection_scores[e];
                        best_idx = e;
                    }
                }
                selected[best_idx] = true;
                all_indices[t][k] = best_idx;
                // BiasFree: use original probs (not augmented) for weights
                all_weights[t * top_k + k] = row[best_idx];
                self.last_expert_counts[best_idx] += 1.0;
            }

            // Renormalize weights
            let sum_w: f32 = all_weights[t * top_k..t * top_k + top_k]
                .iter()
                .sum::<f32>()
                .max(1e-12);
            let inv = 1.0 / sum_w;
            for k in 0..top_k {
                all_weights[t * top_k + k] *= inv;
            }
        }

        self.selection_buf = selection_scores;

        (
            all_indices,
            Tensor::from_vec(all_weights, Shape::new(&[batch_seq, top_k])),
            probs,
        )
    }

    fn route_relu(
        &mut self,
        batch_seq: usize,
        logits_data: &[f32],
        top_k: usize,
    ) -> (Vec<Vec<usize>>, Tensor, Vec<f32>) {
        // ReLU routing: no softmax, apply ReLU to logits
        self.last_logits.clear();
        self.last_logits.extend_from_slice(logits_data);

        let mut all_indices = std::mem::take(&mut self.indices_buf);
        if all_indices.len() != batch_seq {
            all_indices.resize_with(batch_seq, Vec::new);
        }
        for idxs in &mut all_indices {
            if idxs.len() != top_k {
                idxs.resize(top_k, 0);
            }
        }
        let mut all_weights = std::mem::take(&mut self.weights_buf);
        all_weights.resize(batch_seq * top_k, 0.0);
        all_weights.fill(0.0);

        let mut total_active = 0.0;
        let mut relu_sum = 0.0;
        let mut active_buf = std::mem::take(&mut self.active_buf);

        for t in 0..batch_seq {
            let start = t * self.n_experts;
            let logits_row = &logits_data[start..start + self.n_experts];

            // Apply ReLU and collect active experts
            active_buf.clear();
            for (e, &logit) in logits_row.iter().enumerate() {
                if logit > 0.0 {
                    relu_sum += logit;
                    active_buf.push((e, logit));
                }
            }

            // If no active experts, force activate argmax
            if active_buf.is_empty() {
                let argmax = logits_row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                active_buf.push((argmax, 1.0));
            }

            // If more than top_k active, keep only top-k by weight
            if active_buf.len() > top_k {
                active_buf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                active_buf.truncate(top_k);
            }

            total_active += active_buf.len() as f32;

            // Renormalize active weights
            let sum_w: f32 = active_buf.iter().map(|(_, w)| w).sum::<f32>().max(1e-12);
            let inv = 1.0 / sum_w;

            // Fill indices and weights, pad to top_k if needed
            for k in 0..top_k {
                if k < active_buf.len() {
                    all_indices[t][k] = active_buf[k].0;
                    all_weights[t * top_k + k] = active_buf[k].1 * inv;
                } else {
                    // Pad with expert 0, weight 0
                    all_indices[t][k] = 0;
                    all_weights[t * top_k + k] = 0.0;
                }
            }
        }

        self.active_buf = active_buf;

        self.last_avg_active = total_active / batch_seq as f32;
        self.last_relu_sum = relu_sum;

        // Return empty probs for ReLU mode
        (
            all_indices,
            Tensor::from_vec(all_weights, Shape::new(&[batch_seq, top_k])),
            Vec::new(),
        )
    }

    /// Compute router z-loss (ST-MoE) with gradient backprop to gate weights.
    ///
    /// L_z = z_weight * (1/B) * Σ_i logsumexp(logits_i)²
    ///
    /// Gradient: dL_z/d(logits[i,j]) = z_weight * (2/B) * lse_i * softmax(logits_i)_j
    ///
    /// Backprops to gate.weight via: gate_weight_grad += Σ_t grad_logits[t,e] * last_input[t,h]
    pub fn compute_z_loss_with_grad(&mut self, z_weight: f32) -> f32 {
        if self.last_logits.is_empty() || self.last_gate_probs.is_empty() {
            return 0.0;
        }

        let batch_seq = self.last_logits.len() / self.n_experts;
        let mut z_loss_sum = 0.0f32;
        let mut grad_logits = vec![0.0f32; batch_seq * self.n_experts];

        // Compute z-loss and gradients w.r.t. logits
        for t in 0..batch_seq {
            let start = t * self.n_experts;
            let logits_row = &self.last_logits[start..start + self.n_experts];
            let probs_row = &self.last_gate_probs[start..start + self.n_experts];

            // logsumexp with max-subtract trick for numerical stability
            let max_logit = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits_row.iter().map(|&x| (x - max_logit).exp()).sum();
            let lse = max_logit + sum_exp.ln();

            z_loss_sum += lse * lse;

            // Gradient: dL_z/d(logits[t,e]) = z_weight * (2/B) * lse * probs[t,e]
            let grad_scale = z_weight * (2.0 / batch_seq as f32) * lse;
            for e in 0..self.n_experts {
                grad_logits[start + e] = grad_scale * probs_row[e];
            }
        }

        let z_loss = z_weight * z_loss_sum / batch_seq as f32;

        // Backprop grad_logits to gate.weight
        // gate.weight shape: [n_experts, hidden_dim]
        // grad_logits shape: [batch_seq, n_experts]
        // last_input shape: [batch_seq, hidden_dim]
        if let Some(ref last_input) = self.gate.last_input {
            let hidden_dim = self.gate.weight.shape().last_dim();
            let mut grad_weight = vec![0.0f32; self.n_experts * hidden_dim];

            for t in 0..batch_seq {
                for e in 0..self.n_experts {
                    let grad_l = grad_logits[t * self.n_experts + e];
                    for h in 0..hidden_dim {
                        grad_weight[e * hidden_dim + h] += grad_l * last_input[t * hidden_dim + h];
                    }
                }
            }

            // Accumulate into gate.weight.grad
            if let Some(existing_grad) = self.gate.weight.grad_mut() {
                let grad_data = existing_grad.data_mut();
                for (i, &g) in grad_weight.iter().enumerate() {
                    grad_data[i] += g;
                }
            } else {
                self.gate.weight.set_grad(Tensor::from_vec(
                    grad_weight,
                    self.gate.weight.shape().clone(),
                ));
            }
        }

        z_loss
    }

    /// Compute auxiliary load balancing loss with gradient backprop to gate weights.
    ///
    /// aux_loss = alpha * n_experts * Σ_e(f_e * P_e)
    /// where f_e = fraction of tokens assigned to expert e (from last_expert_counts),
    ///       P_e = mean softmax probability for expert e (from last_gate_probs)
    ///
    /// Gradient: dL/d(logits[t,e]) = (alpha * n_experts / B) * probs[t,e] * (f_e - dot(f, probs[t,:]))
    ///
    /// Backprops to gate.weight via: gate_weight_grad += Σ_t grad_logits[t,e] * last_input[t,h]
    pub fn compute_aux_loss_with_grad(&mut self, alpha: f32) -> f32 {
        if self.last_expert_counts.is_empty() || self.last_gate_probs.is_empty() {
            return 0.0;
        }

        let total_assignments: f32 = self.last_expert_counts.iter().sum();
        if total_assignments == 0.0 {
            return 0.0;
        }

        // Compute f_e: fraction of tokens assigned to each expert
        let f_e: Vec<f32> = self
            .last_expert_counts
            .iter()
            .map(|&count| count / total_assignments)
            .collect();

        let batch_seq = self.last_gate_probs.len() / self.n_experts;
        let mut aux_loss_sum = 0.0f32;
        let mut grad_logits = vec![0.0f32; batch_seq * self.n_experts];

        // Compute aux_loss and gradients w.r.t. logits
        for t in 0..batch_seq {
            let start = t * self.n_experts;
            let probs_row = &self.last_gate_probs[start..start + self.n_experts];

            // dot(f_e, probs[t,:]) = Σ_e f_e * probs[t,e]
            let dot_fp_t: f32 = f_e
                .iter()
                .zip(probs_row.iter())
                .map(|(&f, &p)| f * p)
                .sum();

            // Gradient: dL/d(logits[t,e]) = (alpha * n_experts / B) * probs[t,e] * (f_e - dot_fp_t)
            let grad_scale = alpha * self.n_experts as f32 / batch_seq as f32;
            for e in 0..self.n_experts {
                grad_logits[start + e] = grad_scale * probs_row[e] * (f_e[e] - dot_fp_t);
            }

            // Compute aux_loss contribution from this token
            // P_e for this token = probs[t,e]
            for e in 0..self.n_experts {
                aux_loss_sum += f_e[e] * probs_row[e];
            }
        }

        let aux_loss = alpha * self.n_experts as f32 * aux_loss_sum / batch_seq as f32;

        // Backprop grad_logits to gate.weight
        // gate.weight shape: [n_experts, hidden_dim]
        // grad_logits shape: [batch_seq, n_experts]
        // last_input shape: [batch_seq, hidden_dim]
        if let Some(ref last_input) = self.gate.last_input {
            let hidden_dim = self.gate.weight.shape().last_dim();
            let mut grad_weight = vec![0.0f32; self.n_experts * hidden_dim];

            for t in 0..batch_seq {
                for e in 0..self.n_experts {
                    let grad_l = grad_logits[t * self.n_experts + e];
                    for h in 0..hidden_dim {
                        grad_weight[e * hidden_dim + h] += grad_l * last_input[t * hidden_dim + h];
                    }
                }
            }

            // Accumulate into gate.weight.grad
            if let Some(existing_grad) = self.gate.weight.grad_mut() {
                let grad_data = existing_grad.data_mut();
                for (i, &g) in grad_weight.iter().enumerate() {
                    grad_data[i] += g;
                }
            } else {
                self.gate.weight.set_grad(Tensor::from_vec(
                    grad_weight,
                    self.gate.weight.shape().clone(),
                ));
            }
        }

        aux_loss
    }

    /// Returns expert token counts from the last forward pass.
    pub fn expert_counts(&self) -> &[f32] {
        &self.last_expert_counts
    }

    pub fn set_routing_mode(&mut self, mode: RoutingMode) {
        self.routing_mode = mode;
    }

    pub fn update_expert_bias(&mut self, gamma: f32) {
        if self.last_expert_counts.is_empty() {
            return;
        }
        let total_assignments: f32 = self.last_expert_counts.iter().sum();
        if total_assignments == 0.0 {
            return;
        }
        let target = 1.0 / self.n_experts as f32;
        for e in 0..self.n_experts {
            let f_e = self.last_expert_counts[e] / total_assignments;
            let delta = target - f_e;
            self.expert_bias[e] += gamma * delta.signum();
        }
    }

    pub fn set_relu_lambda(&mut self, lambda: f32) {
        self.relu_lambda_l1 = lambda;
    }

    pub fn avg_active_experts(&self) -> f32 {
        self.last_avg_active
    }

    pub fn compute_relu_l1_loss_with_grad(&mut self) -> f32 {
        if self.last_logits.is_empty() {
            return 0.0;
        }

        let batch_seq = self.last_logits.len() / self.n_experts;
        let l1_loss = self.relu_lambda_l1 * self.last_relu_sum / batch_seq as f32;

        // Gradient: lambda * (1/batch_seq) * (1 if logit > 0 else 0)
        let mut grad_logits = vec![0.0f32; batch_seq * self.n_experts];
        let grad_scale = self.relu_lambda_l1 / batch_seq as f32;
        for (i, &logit) in self.last_logits.iter().enumerate() {
            if logit > 0.0 {
                grad_logits[i] = grad_scale;
            }
        }

        // Backprop to gate.weight
        if let Some(ref last_input) = self.gate.last_input {
            let hidden_dim = self.gate.weight.shape().last_dim();
            let mut grad_weight = vec![0.0f32; self.n_experts * hidden_dim];

            for t in 0..batch_seq {
                for e in 0..self.n_experts {
                    let grad_l = grad_logits[t * self.n_experts + e];
                    for h in 0..hidden_dim {
                        grad_weight[e * hidden_dim + h] += grad_l * last_input[t * hidden_dim + h];
                    }
                }
            }

            if let Some(existing_grad) = self.gate.weight.grad_mut() {
                let grad_data = existing_grad.data_mut();
                for (i, &g) in grad_weight.iter().enumerate() {
                    grad_data[i] += g;
                }
            } else {
                self.gate.weight.set_grad(Tensor::from_vec(
                    grad_weight,
                    self.gate.weight.shape().clone(),
                ));
            }
        }

        l1_loss
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
    /// Reusable buffer for forward output accumulator (avoids alloc per forward)
    output_buf: Vec<f32>,
    /// Reusable buffer for per-expert token gather (avoids alloc per expert)
    gather_buf: Vec<f32>,
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
            output_buf: Vec::new(),
            gather_buf: Vec::new(),
        }
    }

    pub(crate) fn router(&self) -> &Router {
        &self.router
    }

    pub(crate) fn router_mut(&mut self) -> &mut Router {
        &mut self.router
    }

    pub(crate) fn experts(&self) -> &[ExpertFFN] {
        &self.experts
    }

    pub(crate) fn experts_mut(&mut self) -> &mut [ExpertFFN] {
        &mut self.experts
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn set_routing_mode(&mut self, mode: RoutingMode) {
        self.router.set_routing_mode(mode);
    }

    pub fn update_expert_bias(&mut self, gamma: f32) {
        self.router.update_expert_bias(gamma);
    }

    pub fn set_relu_lambda(&mut self, lambda: f32) {
        self.router.set_relu_lambda(lambda);
    }

    pub fn avg_active_experts(&self) -> f32 {
        self.router.avg_active_experts()
    }

    pub fn compute_relu_l1_loss_with_grad(&mut self) -> f32 {
        self.router.compute_relu_l1_loss_with_grad()
    }

    /// Reclaim the output buffer from a consumed forward Tensor.
    /// Call after using the forward result to recycle the allocation.
    pub fn reclaim_output(&mut self, tensor: Tensor) {
        self.output_buf = tensor.into_data();
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
        let mut out = std::mem::take(&mut self.output_buf);
        out.resize(batch_seq * hidden, 0.0);
        out.fill(0.0);

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
        let mut gather_buf = std::mem::take(&mut self.gather_buf);
        for e_idx in 0..n_experts {
            let tokens = &self.expert_tokens_buf[e_idx];
            if tokens.is_empty() {
                continue;
            }
            let n_tok = tokens.len();

            // Gather: collect assigned token vectors into a contiguous batch
            gather_buf.resize(n_tok * hidden, 0.0);
            for (i, &t) in tokens.iter().enumerate() {
                let src = &input_data[t * hidden..(t + 1) * hidden];
                gather_buf[i * hidden..(i + 1) * hidden].copy_from_slice(src);
            }

            // Single batched expert forward (1 BLAS call instead of n_tok calls)
            let batch_input = Tensor::from_vec(
                std::mem::take(&mut gather_buf),
                Shape::new(&[n_tok, 1, hidden]),
            );
            let expert_out = self.experts[e_idx].forward(&batch_input);
            gather_buf = batch_input.into_data();
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
        self.gather_buf = gather_buf;

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
            self.router.weights_buf = weights.into_data();
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

    pub(crate) fn attn_norm(&self) -> &RMSNorm {
        &self.attn_norm
    }

    pub(crate) fn attention(&self) -> &MQAttention {
        &self.attention
    }

    pub(crate) fn attention_mut(&mut self) -> &mut MQAttention {
        &mut self.attention
    }

    pub(crate) fn ffn_norm(&self) -> &RMSNorm {
        &self.ffn_norm
    }

    pub(crate) fn moe(&self) -> &MoELayer {
        &self.moe
    }

    pub(crate) fn moe_mut(&mut self) -> &mut MoELayer {
        &mut self.moe
    }

    pub fn set_routing_mode(&mut self, mode: RoutingMode) {
        self.moe.set_routing_mode(mode);
    }

    pub fn update_routing_biases(&mut self, gamma: f32) {
        self.moe.update_expert_bias(gamma);
    }

    pub fn set_relu_lambda(&mut self, lambda: f32) {
        self.moe.set_relu_lambda(lambda);
    }

    pub fn avg_active_experts(&self) -> f32 {
        self.moe.avg_active_experts()
    }

    pub fn apply_relu_l1_loss(&mut self) -> f32 {
        self.moe.compute_relu_l1_loss_with_grad()
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
