// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Top-level MoE Transformer model.
//!
//! Architecture:
//!   Embedding -> [TransformerBlock x n_layers] -> RMSNorm -> Linear (lm_head)
//!
//! Forward path (forward_ids):
//!   1. Token IDs -> Embedding lookup -> [batch, seq, hidden]
//!   2. Pass through each TransformerBlock (attention + MoE FFN with residuals)
//!   3. Final RMSNorm -> lm_head projection -> [batch, seq, vocab] logits
//!
//! The model also provides convenience methods for text generation with
//! different sampling strategies (greedy, temperature, top-k, top-p).

use crate::config::Config;
use crate::generate::{self, SamplingStrategy};
use crate::layers::{Embedding, Layer, Linear, RMSNorm};
use crate::moe::TransformerBlock;
use crate::tensor::Tensor;

/// Full MoE Transformer language model.
///
/// Owns all parameters through its sub-layers. Parameter traversal via
/// the Layer trait collects references from all blocks without cloning.
pub struct MoETransformer {
    config: Config,
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RMSNorm,
    /// Projects hidden states to vocabulary logits (no softmax -- raw logits)
    lm_head: Linear,
}

impl MoETransformer {
    pub fn new(config: Config) -> Self {
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config))
            .collect();
        Self {
            embedding: Embedding::new(config.vocab_size, config.hidden_dim),
            final_norm: RMSNorm::new(config.hidden_dim),
            lm_head: Linear::new(config.hidden_dim, config.vocab_size),
            blocks,
            config,
        }
    }

    pub fn default_6_9b() -> Self {
        Self::new(Config::default_6_9b())
    }

    pub fn tiny() -> Self {
        Self::new(Config::tiny())
    }

    pub fn small() -> Self {
        Self::new(Config::small())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_inference_mode(&mut self, mode: bool) {
        for block in &mut self.blocks {
            block.set_inference_mode(mode);
        }
        self.final_norm.set_inference_mode(mode);
        self.lm_head.set_inference_mode(mode);
    }

    pub fn total_aux_loss(&self, alpha: f32) -> f32 {
        self.blocks.iter().map(|blk| blk.moe.aux_loss(alpha)).sum()
    }

    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Full forward pass from token IDs to logits.
    ///
    /// Pipeline: embed -> blocks[0..n] -> final_norm -> lm_head
    /// Output shape: [batch, seq_len, vocab_size]
    ///
    /// Each block.forward() consumes the previous `x` and produces a new tensor.
    /// `x` is moved (not cloned) into the block, and the returned value replaces it.
    /// This means only one hidden-state tensor is live at a time per layer.
    pub fn forward_ids(&mut self, token_ids: &[usize], batch: usize, seq_len: usize) -> Tensor {
        let mut x = self.embedding.forward_with_ids(token_ids, batch, seq_len);
        for block in &mut self.blocks {
            x = block.forward(&x);
        }
        let normed = self.final_norm.forward(&x);
        self.lm_head.forward(&normed)
    }

    pub fn generate_greedy(&mut self, prompt: &[usize], max_len: usize) -> Vec<usize> {
        generate::generate(self, prompt, max_len, &SamplingStrategy::Greedy, 0)
    }

    pub fn generate_sample(
        &mut self,
        prompt: &[usize],
        max_len: usize,
        temperature: f32,
        seed: u64,
    ) -> Vec<usize> {
        generate::generate(
            self,
            prompt,
            max_len,
            &SamplingStrategy::Sample { temperature },
            seed,
        )
    }

    pub fn generate_top_k(
        &mut self,
        prompt: &[usize],
        max_len: usize,
        k: usize,
        temperature: f32,
        seed: u64,
    ) -> Vec<usize> {
        generate::generate(
            self,
            prompt,
            max_len,
            &SamplingStrategy::TopK { k, temperature },
            seed,
        )
    }

    pub fn generate_top_p(
        &mut self,
        prompt: &[usize],
        max_len: usize,
        top_p: f32,
        temperature: f32,
        seed: u64,
    ) -> Vec<usize> {
        generate::generate(
            self,
            prompt,
            max_len,
            &SamplingStrategy::TopP { top_p, temperature },
            seed,
        )
    }
}

impl Layer for MoETransformer {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let dims = input.shape().dims();
        let token_ids: Vec<usize> = input.data().iter().map(|&x| x as usize).collect();
        self.forward_ids(&token_ids, dims[0], dims[1])
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = self.lm_head.backward(grad_output);
        grad = self.final_norm.backward(&grad);
        for block in self.blocks.iter_mut().rev() {
            grad = block.backward(&grad);
        }
        // Embedding backward: accumulates weight gradients via scatter-add
        self.embedding.backward(&grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.embedding.parameters();
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.final_norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.embedding.parameters_mut();
        for block in &mut self.blocks {
            params.extend(block.parameters_mut());
        }
        params.extend(self.final_norm.parameters_mut());
        params.extend(self.lm_head.parameters_mut());
        params
    }
}
