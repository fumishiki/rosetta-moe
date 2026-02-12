// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Layer trait and basic layer implementations.
//!
//! Building blocks: Embedding, RMSNorm, Linear, SwiGLU.
//! Each layer implements the [`Layer`] trait (forward, backward, parameters).
//!
//! Helper macros reduce boilerplate for parameter collection:
//! - `collect_params!` / `collect_params_mut!` -- flatten params from sub-layers
//! - `impl_single_weight_params!` -- for layers with exactly one weight tensor

use crate::tensor::{DType, Shape, Tensor};

// ---- Parameter collection macros ----
// These exist because Rust doesn't have trait-based automatic parameter traversal
// like PyTorch's nn.Module. Each composite layer must manually aggregate its
// sub-layers' parameters for the optimizer.

macro_rules! collect_params {
    ($($layer:expr),+ $(,)?) => {{
        let mut params = Vec::new();
        $(params.extend($layer.parameters());)+
        params
    }};
}

macro_rules! collect_params_mut {
    ($($layer:expr),+ $(,)?) => {{
        let mut params = Vec::new();
        $(params.extend($layer.parameters_mut());)+
        params
    }};
}

macro_rules! impl_single_weight_params {
    ($field:ident) => {
        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.$field]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.$field]
        }
    };
}

pub(crate) use collect_params;
pub(crate) use collect_params_mut;

pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn set_inference_mode(&mut self, _mode: bool) {}
}

/// Accumulate gradient into a parameter's grad field.
/// If grad already exists (zeroed by clear_grad), add in-place.
/// If grad is None (first ever call), allocate and copy.
fn accumulate_grad(param: &mut Tensor, grad: &[f32]) {
    if let Some(existing) = param.grad_mut() {
        let data = existing.data_mut();
        for (d, g) in data.iter_mut().zip(grad.iter()) {
            *d += g;
        }
    } else {
        param.set_grad(Tensor::from_vec(grad.to_vec(), param.shape().clone()));
    }
}

/// Embedding lookup table: token_id -> hidden vector.
///
/// Embedding(token) = W[token, :] where W is [vocab_size, hidden_dim].
/// Initialization: N(0, sqrt(2/hidden_dim)) (He initialization).
pub struct Embedding {
    pub weight: Tensor,
    /// Cached token IDs from the last forward pass for backward.
    last_ids: Option<Vec<usize>>,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Self {
        // He init: std = sqrt(2/fan_in) where fan_in = hidden_dim
        let std = (2.0 / hidden_dim as f32).sqrt();
        Self {
            weight: Tensor::randn(Shape::new(&[vocab_size, hidden_dim]), DType::F32, 42).scale(std),
            last_ids: None,
        }
    }

    /// Direct integer-indexed embedding lookup (avoids casting floats to indices).
    /// Each token_id selects one row from the weight matrix via copy_from_slice.
    /// Uses from_vec (zero-copy construction) since `out` is freshly allocated here.
    pub fn forward_with_ids(
        &mut self,
        token_ids: &[usize],
        batch: usize,
        seq_len: usize,
    ) -> Tensor {
        let hidden_dim = self.weight.shape().last_dim();
        let mut out = vec![0.0; batch * seq_len * hidden_dim];

        for (i, &token) in token_ids.iter().enumerate() {
            let src = token * hidden_dim;
            let dst = i * hidden_dim;
            out[dst..dst + hidden_dim].copy_from_slice(&self.weight.data()[src..src + hidden_dim]);
        }

        // Cache token IDs for backward
        self.last_ids = Some(token_ids.to_vec());

        Tensor::from_vec(out, Shape::new(&[batch, seq_len, hidden_dim]))
    }
}

impl Layer for Embedding {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let dims = input.shape().dims();
        let token_ids: Vec<usize> = input.data().iter().map(|&x| x as usize).collect();
        self.forward_with_ids(&token_ids, dims[0], dims[1])
    }

    /// Backward: scatter-add grad_output into weight.grad at token indices.
    /// Returns zeros (no meaningful gradient for discrete token IDs).
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        if let Some(ref ids) = self.last_ids {
            let hidden_dim = self.weight.shape().last_dim();
            let vocab_size = self.weight.shape().dims()[0];
            let mut grad_w = vec![0.0f32; vocab_size * hidden_dim];
            let grad_data = grad_output.data();

            // Scatter-add: for each token position, accumulate grad into the corresponding row
            for (i, &token_id) in ids.iter().enumerate() {
                let src = i * hidden_dim;
                let dst = token_id * hidden_dim;
                for j in 0..hidden_dim {
                    grad_w[dst + j] += grad_data[src + j];
                }
            }

            accumulate_grad(&mut self.weight, &grad_w);
        }
        // Return zeros shaped like grad_output (no meaningful gradient for discrete token IDs)
        Tensor::zeros(grad_output.shape().clone(), DType::F32)
    }

    impl_single_weight_params!(weight);
}

/// RMSNorm (Root Mean Square Layer Normalization).
///
/// RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma
///
/// Unlike LayerNorm, RMSNorm does not center (subtract mean), only scales.
/// This saves one reduction pass and is standard in LLaMA-family models.
/// `gamma` (the learnable scale) is initialized to all 1s.
pub struct RMSNorm {
    pub weight: Tensor,
    eps: f32,
    /// Cached input and inv_rms from forward for backward.
    last_input: Option<Vec<f32>>,
    last_inv_rms: Option<Vec<f32>>,
    last_hidden: usize,
    /// Pre-allocated backward scratch buffers (reused across steps)
    grad_gamma_buf: Option<Vec<f32>>,
    grad_input_buf: Option<Vec<f32>>,
    inference_mode: bool,
}

impl RMSNorm {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            weight: Tensor::ones(Shape::new(&[hidden_dim]), DType::F32),
            eps: 1e-6,
            last_input: None,
            last_inv_rms: None,
            last_hidden: 0,
            grad_gamma_buf: None,
            grad_input_buf: None,
            inference_mode: false,
        }
    }
}

impl Layer for RMSNorm {
    // RMSNorm: y_j = x_j * (1 / sqrt(mean(x^2) + eps)) * gamma_j
    // Operates row-wise over the last dimension (hidden_dim).
    //
    // Uses SIMD approximate rsqrt (NEON vrsqrteq_f32 + Newton-Raphson) for
    // the 1/sqrt(mean_sq + eps) computation, matching Julia's @fastmath strategy.
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let hidden = input.shape().last_dim();
        let outer = input.shape().batch_size();
        let mut out = vec![0.0; input.numel()];
        let mut inv_rms_vec = Vec::with_capacity(outer);

        // Reusable single-element buffer for SIMD rsqrt (avoids per-row allocation)
        let mut rsqrt_in = [0.0f32; 1];
        let mut rsqrt_out = [0.0f32; 1];

        for row_idx in 0..outer {
            let start = row_idx * hidden;
            let row = &input.data()[start..start + hidden];
            // mean(x^2) = sum(x_i^2) / d
            let mean_sq = row.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
            // inv_rms = approx 1 / sqrt(mean(x^2) + eps) via SIMD rsqrt
            rsqrt_in[0] = mean_sq;
            crate::simd::fast_rsqrt_slice(&rsqrt_in, self.eps, &mut rsqrt_out);
            let inv_rms = rsqrt_out[0];
            inv_rms_vec.push(inv_rms);

            for (j, &x) in row.iter().enumerate() {
                out[start + j] = x * inv_rms * self.weight.data()[j];
            }
        }

        // Cache for backward
        if !self.inference_mode {
            self.last_input = Some(input.data().to_vec());
            self.last_inv_rms = Some(inv_rms_vec);
            self.last_hidden = hidden;
        }

        Tensor::from_vec(out, input.shape().clone())
    }

    /// Backward for RMSNorm:
    ///   y = (x / rms) * gamma
    ///   grad_gamma = sum(grad_output * x / rms, over batch dims)
    ///   grad_input = grad_output * gamma / rms  (simplified)
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let hidden = self.last_hidden;
        let go = grad_output.data();

        if let (Some(input_data), Some(inv_rms_vec)) = (&self.last_input, &self.last_inv_rms) {
            let outer = inv_rms_vec.len();
            let gamma = self.weight.data();

            // Compute grad_gamma = sum over batch of (grad_output * x * inv_rms)
            let mut grad_gamma = self.grad_gamma_buf.take().unwrap_or_default();
            grad_gamma.resize(hidden, 0.0);
            grad_gamma.fill(0.0);
            for row_idx in 0..outer {
                let start = row_idx * hidden;
                let inv_rms = inv_rms_vec[row_idx];
                for j in 0..hidden {
                    // x_norm = x * inv_rms
                    grad_gamma[j] += go[start + j] * input_data[start + j] * inv_rms;
                }
            }

            // Compute grad_input = grad_output * gamma * inv_rms - x * dot_term / (dim * rms^3)
            let mut grad_input = self.grad_input_buf.take().unwrap_or_default();
            grad_input.resize(go.len(), 0.0);
            grad_input.fill(0.0);
            for row_idx in 0..outer {
                let start = row_idx * hidden;
                let inv_rms = inv_rms_vec[row_idx];
                let rms3 = 1.0 / (inv_rms * inv_rms * inv_rms);
                let inv_dim_rms3 = 1.0 / (hidden as f32 * rms3);

                // Compute dot_sum = sum(grad_output * gamma * input * inv_rms)
                let mut dot_sum = 0.0f32;
                for j in 0..hidden {
                    dot_sum += go[start + j] * gamma[j] * input_data[start + j] * inv_rms;
                }

                // Apply full gradient formula with Jacobian correction
                for j in 0..hidden {
                    grad_input[start + j] = go[start + j] * gamma[j] * inv_rms
                        - input_data[start + j] * dot_sum * inv_dim_rms3;
                }
            }

            // Accumulate grad_gamma after gamma borrow is done
            accumulate_grad(&mut self.weight, &grad_gamma);
            self.grad_gamma_buf = Some(grad_gamma);

            Tensor::from_vec(grad_input, grad_output.shape().clone())
        } else {
            grad_output.clone()
        }
    }

    impl_single_weight_params!(weight);

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
    }
}

/// Fully connected linear layer (no bias).
///
/// Forward:  Y = X @ W^T   where W is [out_features, in_features]
/// Backward: dX = dY @ W   (gradient w.r.t. input)
///           dW = dY^T @ X  (gradient w.r.t. weight)
///
/// Weight shape is [out, in] (PyTorch convention) so forward uses sgemm_transb
/// to compute X @ W^T without an explicit transpose.
pub struct Linear {
    pub weight: Tensor,
    /// Cached flattened input from forward for backward weight gradient.
    pub(crate) last_input: Option<Vec<f32>>,
    pub(crate) last_batch: usize,
    /// Pre-allocated backward scratch buffers (reused across steps)
    dx_buf: Option<Vec<f32>>,
    dw_buf: Option<Vec<f32>>,
    inference_mode: bool,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // He init: std = sqrt(2/fan_in)
        let std = (2.0 / in_features as f32).sqrt();
        Self {
            weight: Tensor::randn(Shape::new(&[out_features, in_features]), DType::F32, 123)
                .scale(std),
            last_input: None,
            last_batch: 0,
            dx_buf: None,
            dw_buf: None,
            inference_mode: false,
        }
    }
}

impl Layer for Linear {
    // Y = X @ W^T  via sgemm_transb (avoids materializing the transpose).
    // X: [batch, in_features], W: [out_features, in_features], Y: [batch, out_features]
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let out_features = self.weight.shape().dims()[0];
        let in_features = self.weight.shape().last_dim();
        let batch = input.shape().batch_size();

        // Cache input for backward
        if !self.inference_mode {
            self.last_input = Some(input.data().to_vec());
        }
        self.last_batch = batch;

        let mut out = vec![0.0; batch * out_features];
        // C = A @ B^T: A=input[batch, in], B=W[out, in], C=output[batch, out]
        crate::accelerate::sgemm_transb(
            batch,
            out_features,
            in_features,
            1.0,
            input.data(),
            self.weight.data(),
            0.0,
            &mut out,
        );

        // Build output shape: replace last dim of input shape with out_features
        let out_shape = input.shape().with_last_dim(out_features);
        Tensor::from_vec(out, out_shape)
    }

    // dX = dY @ W (no transpose -- W is already [out, in], so dY[batch, out] @ W[out, in])
    // dW = dY^T @ X   shape: [out_features, in_features]
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let out_features = self.weight.shape().dims()[0];
        let in_features = self.weight.shape().last_dim();
        let batch = grad_output.shape().batch_size();

        // dX = dY @ W (reuse pre-allocated buffer)
        let dx_len = batch * in_features;
        let mut dx = self.dx_buf.take().unwrap_or_default();
        dx.resize(dx_len, 0.0);
        dx.fill(0.0);
        crate::accelerate::sgemm(
            batch,
            in_features,
            out_features,
            1.0,
            grad_output.data(),
            self.weight.data(),
            0.0,
            &mut dx,
        );

        // dW = dY^T @ X  (sgemm_transa: A^T @ B where A=[batch, out], B=[batch, in])
        // Result: [out_features, in_features]
        if let Some(ref input_data) = self.last_input {
            let dw_len = out_features * in_features;
            let mut dw = self.dw_buf.take().unwrap_or_default();
            dw.resize(dw_len, 0.0);
            dw.fill(0.0);
            crate::accelerate::sgemm_transa(
                out_features,
                in_features,
                batch,
                1.0,
                grad_output.data(),
                input_data,
                0.0,
                &mut dw,
            );
            accumulate_grad(&mut self.weight, &dw);
            self.dw_buf = Some(dw);
        }

        let out_shape = grad_output.shape().with_last_dim(in_features);
        Tensor::from_vec(dx, out_shape)
    }

    impl_single_weight_params!(weight);

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
    }
}

/// SwiGLU feed-forward network (Shazeer 2020).
///
/// SwiGLU(x) = Down(SiLU(Gate(x)) * Up(x))
///
/// Expanded:
///   gate = W_gate @ x                    [hidden -> ffn_dim]
///   up   = W_up   @ x                    [hidden -> ffn_dim]
///   h    = silu(gate) * up               element-wise gating
///   out  = W_down @ h                    [ffn_dim -> hidden]
///
/// SiLU(x) = x * sigmoid(x). The gating mechanism allows the network
/// to learn which dimensions to activate, improving over standard ReLU FFNs.
pub struct SwiGLU {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
    /// Cached activations for backward
    pub(crate) last_gate_pre_silu: Option<Vec<f32>>,
    pub(crate) last_up_out: Option<Vec<f32>>,
    grad_gate_buf: Option<Vec<f32>>,
    grad_up_buf: Option<Vec<f32>>,
    inference_mode: bool,
}

impl SwiGLU {
    pub fn new(hidden_dim: usize, ffn_dim: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_dim, ffn_dim),
            up_proj: Linear::new(hidden_dim, ffn_dim),
            down_proj: Linear::new(ffn_dim, hidden_dim),
            last_gate_pre_silu: None,
            last_up_out: None,
            grad_gate_buf: None,
            grad_up_buf: None,
            inference_mode: false,
        }
    }
}

impl Layer for SwiGLU {
    // SwiGLU: output = Down(SiLU(Gate(x)) * Up(x))
    // Caches pre-silu gate and up for backward.
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut gate = self.gate_proj.forward(input);
        if !self.inference_mode {
            let mut gate_pre = self.last_gate_pre_silu.take().unwrap_or_default();
            gate_pre.clear();
            gate_pre.extend_from_slice(gate.data());
            self.last_gate_pre_silu = Some(gate_pre);
        }
        gate.silu_in_place();

        let up = self.up_proj.forward(input);
        gate.mul_in_place(&up);
        if !self.inference_mode {
            // Move expert up-projection output after use to avoid a clone.
            self.last_up_out = Some(up.into_data());
        }
        self.down_proj.forward(&gate)
    }

    /// Backward for SwiGLU:
    ///   grad_hidden = down.backward(grad_output)
    ///   grad_silu_gate = grad_hidden * up_out
    ///   grad_up_out = grad_hidden * silu(gate)
    ///   grad_gate = grad_silu_gate * silu'(gate_pre_silu)
    ///     where silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    ///   grad_input = gate_proj.backward(grad_gate) + up_proj.backward(grad_up_out)
    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Backward through down_proj
        let grad_hidden = self.down_proj.backward(grad_output);
        let gh = grad_hidden.data();

        let gate_pre_silu = self.last_gate_pre_silu.take().unwrap_or_default();
        let up_out = self.last_up_out.take().unwrap_or_default();

        let n = gh.len();
        let mut grad_gate = self.grad_gate_buf.take().unwrap_or_default();
        grad_gate.resize(n, 0.0);
        let mut grad_up = self.grad_up_buf.take().unwrap_or_default();
        grad_up.resize(n, 0.0);

        for i in 0..n {
            // grad_up_out = grad_hidden * silu(gate_pre_silu)
            let z = gate_pre_silu[i];
            let sig = 1.0 / (1.0 + (-z).exp());
            let silu = z * sig;
            grad_up[i] = gh[i] * silu;

            // grad_silu_gate = grad_hidden * up_out, then chain with silu'(z)
            let grad_silu_out = gh[i] * up_out[i];
            let dsilu = sig * (1.0 + z * (1.0 - sig));
            grad_gate[i] = grad_silu_out * dsilu;
        }

        let grad_gate_tensor = Tensor::from_vec(grad_gate, grad_hidden.shape().clone());
        let grad_up_tensor = Tensor::from_vec(grad_up, grad_hidden.shape().clone());

        // Backward through gate and up projections (accumulates their weight gradients)
        let mut grad_x_gate = self.gate_proj.backward(&grad_gate_tensor);
        let grad_x_up = self.up_proj.backward(&grad_up_tensor);
        self.grad_gate_buf = Some(grad_gate_tensor.into_data());
        self.grad_up_buf = Some(grad_up_tensor.into_data());

        // Sum gradients from both paths (in-place to avoid allocation)
        grad_x_gate.add_in_place(&grad_x_up);
        grad_x_gate
    }

    fn parameters(&self) -> Vec<&Tensor> {
        collect_params!(self.gate_proj, self.up_proj, self.down_proj)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        collect_params_mut!(self.gate_proj, self.up_proj, self.down_proj)
    }

    fn set_inference_mode(&mut self, mode: bool) {
        self.inference_mode = mode;
        self.gate_proj.set_inference_mode(mode);
        self.up_proj.set_inference_mode(mode);
        self.down_proj.set_inference_mode(mode);
    }
}

/// Each MoE expert is a SwiGLU FFN (same architecture, independent weights).
pub type ExpertFFN = SwiGLU;
