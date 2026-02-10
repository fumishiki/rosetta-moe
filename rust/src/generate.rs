// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Token generation with sampling strategies.
//!
//! Autoregressive generation: at each step, run full forward pass on all
//! tokens so far, take the logits for the last position, sample a token,
//! append it, and repeat. (No KV cache -- educational implementation.)
//!
//! Sampling strategies:
//! - **Greedy**: argmax(logits)
//! - **Temperature**: softmax(logits / T), then sample from distribution
//! - **Top-k**: zero out all but the top-k logits, then temperature sample
//! - **Top-p (nucleus)**: keep smallest set of tokens with cumulative prob >= p

use std::cmp::Ordering;

use crate::model::MoETransformer;
use crate::tensor;

/// Sampling strategy for token generation.
pub enum SamplingStrategy {
    /// Always pick the highest-probability token
    Greedy,
    /// Sample from softmax(logits / temperature)
    Sample { temperature: f32 },
    /// Keep only top-k logits, then temperature sample
    TopK { k: usize, temperature: f32 },
    /// Keep tokens until cumulative probability >= top_p, then temperature sample
    TopP { top_p: f32, temperature: f32 },
}

pub fn pick_token(logits: &[f32], strategy: &SamplingStrategy, state: &mut u64) -> usize {
    match *strategy {
        SamplingStrategy::Greedy => argmax(logits),
        SamplingStrategy::Sample { temperature } => sample_from_logits(logits, temperature, state),
        SamplingStrategy::TopK { k, temperature } => sample_top_k(logits, k, temperature, state),
        SamplingStrategy::TopP { top_p, temperature } => {
            sample_top_p(logits, top_p, temperature, state)
        }
    }
}

/// Autoregressive generation loop.
///
/// No KV cache: re-runs the full sequence through the model at each step.
/// This is O(n^2 * model_cost) but simple and correct for educational purposes.
///
/// Logits layout: [1, seq_len, vocab_size] flattened. The last position's logits
/// are at offset (seq_len-1)*vocab .. seq_len*vocab.
pub fn generate(
    model: &mut MoETransformer,
    prompt: &[usize],
    max_len: usize,
    strategy: &SamplingStrategy,
    seed: u64,
) -> Vec<usize> {
    let mut tokens = if prompt.is_empty() {
        vec![0]
    } else {
        prompt.to_vec()
    };
    let mut state = seed;
    let vocab = model.config().vocab_size;

    while tokens.len() < max_len {
        // Full forward pass on entire sequence (no KV cache)
        let logits = model.forward_ids(&tokens, 1, tokens.len());
        // Extract logits for the last position only
        let last = &logits.data()[(tokens.len() - 1) * vocab..tokens.len() * vocab];
        tokens.push(pick_token(last, strategy, &mut state));
    }

    tokens
}

pub fn generate_greedy(model: &mut MoETransformer, prompt: &[usize], max_len: usize) -> Vec<usize> {
    generate(model, prompt, max_len, &SamplingStrategy::Greedy, 0)
}

pub fn generate_sample(
    model: &mut MoETransformer,
    prompt: &[usize],
    max_len: usize,
    temperature: f32,
    seed: u64,
) -> Vec<usize> {
    generate(
        model,
        prompt,
        max_len,
        &SamplingStrategy::Sample { temperature },
        seed,
    )
}

pub fn generate_top_k(
    model: &mut MoETransformer,
    prompt: &[usize],
    max_len: usize,
    k: usize,
    temperature: f32,
    seed: u64,
) -> Vec<usize> {
    generate(
        model,
        prompt,
        max_len,
        &SamplingStrategy::TopK { k, temperature },
        seed,
    )
}

pub fn generate_top_p(
    model: &mut MoETransformer,
    prompt: &[usize],
    max_len: usize,
    top_p: f32,
    temperature: f32,
    seed: u64,
) -> Vec<usize> {
    generate(
        model,
        prompt,
        max_len,
        &SamplingStrategy::TopP { top_p, temperature },
        seed,
    )
}

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn sorted_indices_desc(values: &[f32]) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..values.len()).collect();
    idxs.sort_unstable_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
    idxs
}

/// Temperature sampling: softmax(logits / T), then categorical sample.
///
/// p_i = exp(logit_i / T) / sum_j(exp(logit_j / T))
/// T -> 0: approaches greedy. T -> inf: approaches uniform.
fn sample_from_logits(logits: &[f32], temperature: f32, state: &mut u64) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    // Multiply by 1/T instead of dividing each element (one division vs N)
    let inv_temp = 1.0 / temperature;
    let mut probs: Vec<f32> = logits.iter().map(|x| x * inv_temp).collect();
    tensor::softmax_in_place(&mut probs);
    sample_from_probs(&probs, state)
}

/// Top-k sampling: keep only the top-k logits, set rest to -inf, then sample.
/// Setting non-top-k logits to -inf means softmax gives them probability 0.
fn sample_top_k(logits: &[f32], k: usize, temperature: f32, state: &mut u64) -> usize {
    if k == 0 || k >= logits.len() {
        return sample_from_logits(logits, temperature, state);
    }
    let idxs = sorted_indices_desc(logits);
    let mut filtered = vec![f32::NEG_INFINITY; logits.len()];
    for &idx in &idxs[..k] {
        filtered[idx] = logits[idx];
    }
    sample_from_logits(&filtered, temperature, state)
}

/// Top-p (nucleus) sampling: keep the smallest set of tokens whose cumulative
/// probability exceeds top_p, then renormalize and sample.
///
/// Adaptive: high-confidence predictions keep fewer tokens (sharper distribution),
/// uncertain predictions keep more (flatter distribution).
fn sample_top_p(logits: &[f32], top_p: f32, temperature: f32, state: &mut u64) -> usize {
    if !(0.0..1.0).contains(&top_p) {
        return sample_from_logits(logits, temperature, state);
    }
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let inv_temp = 1.0 / temperature;
    let mut probs: Vec<f32> = logits.iter().map(|x| x * inv_temp).collect();
    tensor::softmax_in_place(&mut probs);

    // Sort by probability descending, accumulate until >= top_p
    let idxs = sorted_indices_desc(&probs);
    let mut kept = Vec::new();
    let mut mass = 0.0;
    for idx in idxs {
        kept.push(idx);
        mass += probs[idx];
        if mass >= top_p {
            break;
        }
    }

    // Renormalize the kept tokens and sample
    let mut truncated: Vec<f32> = kept.iter().map(|&idx| probs[idx]).collect();
    tensor::normalize_in_place(&mut truncated);
    kept[sample_from_probs(&truncated, state)]
}

/// Categorical sampling via inverse CDF: draw r ~ U(0,1), walk the CDF
/// until cumulative probability exceeds r. Falls back to last token if
/// floating-point rounding causes the walk to not trigger.
fn sample_from_probs(probs: &[f32], state: &mut u64) -> usize {
    let r = next_rand01(state);
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += *p;
        if r <= cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// LCG-based PRNG returning a float in [0, 1).
/// Uses the upper 32 bits of the 64-bit state for better quality than lower bits.
fn next_rand01(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let hi = (*state >> 32) as u32;
    hi as f32 / 4294967296.0
}
