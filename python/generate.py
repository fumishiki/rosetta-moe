# SPDX-License-Identifier: CC-BY-NC-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Sampling strategies and token generation for MoE Transformer.

Provides four sampling strategies:
  - Greedy (argmax)
  - Temperature scaling
  - Top-k filtering
  - Top-p (nucleus) filtering

The generate() function runs autoregressive decoding: at each step it
feeds the full token sequence through the model (no KV cache), takes
the logits at the last position, samples a token, and appends it.

All random sampling uses a deterministic LCG PRNG for reproducibility
across platforms (independent of numpy's RNG state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""

    @abstractmethod
    def pick_token(self, logits: np.ndarray) -> int:
        """Pick the next token given logits [vocab_size]."""
        ...


class GreedySampling(SamplingStrategy):
    """Greedy (argmax) sampling — always picks the highest-probability token."""

    def pick_token(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))


class TemperatureSampling(SamplingStrategy):
    """Temperature-based sampling.

    Sampling: p_i = softmax(logits_i / temperature)
    Higher temperature -> more uniform distribution (more random).
    Lower temperature -> sharper distribution (more deterministic).
    """

    def __init__(self, temperature: float = 1.0, seed: int = 42):
        self.temperature = temperature
        self._state = [seed]

    def pick_token(self, logits: np.ndarray) -> int:
        return _sample_from_logits(logits, self.temperature, self._state)


class TopKSampling(SamplingStrategy):
    """Top-k sampling.

    Filters to the k highest-probability tokens before sampling.
    All other tokens are masked to -inf so they receive zero probability.
    """

    def __init__(self, k: int = 50, temperature: float = 1.0, seed: int = 42):
        self.k = k
        self.temperature = temperature
        self._state = [seed]

    def pick_token(self, logits: np.ndarray) -> int:
        return _sample_top_k(logits, self.k, self.temperature, self._state)


class TopPSampling(SamplingStrategy):
    """Top-p (nucleus) sampling.

    Sorts tokens by probability and includes just enough tokens for
    their cumulative probability to exceed top_p.  This dynamically
    adjusts the candidate set size based on the distribution shape.
    """

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0, seed: int = 42):
        self.top_p = top_p
        self.temperature = temperature
        self._state = [seed]

    def pick_token(self, logits: np.ndarray) -> int:
        return _sample_top_p(logits, self.top_p, self.temperature, self._state)


def generate(model, prompt: list[int], max_len: int, strategy: SamplingStrategy) -> list[int]:
    """Generate tokens autoregressively using a given sampling strategy.

    At each step, the full sequence is re-processed through the model
    (no KV cache), which is O(n^2) in sequence length.  This is
    intentionally simple for educational purposes.
    """
    tokens = list(prompt) if prompt else [0]
    while len(tokens) < max_len:
        logits = model.forward_ids(tokens, 1, len(tokens))
        # Take logits at the last position for next-token prediction
        last = logits.data[0, len(tokens) - 1]
        tokens.append(strategy.pick_token(last))
    return tokens


# Backward-compatible function wrappers

def generate_greedy(model, prompt: list[int], max_len: int) -> list[int]:
    """Generate tokens using greedy decoding."""
    return generate(model, prompt, max_len, GreedySampling())


def generate_sample(
    model, prompt: list[int], max_len: int, temperature: float = 1.0, seed: int = 42
) -> list[int]:
    """Generate tokens using temperature sampling."""
    return generate(model, prompt, max_len, TemperatureSampling(temperature, seed))


def generate_top_k(
    model,
    prompt: list[int],
    max_len: int,
    k: int = 50,
    temperature: float = 1.0,
    seed: int = 42,
) -> list[int]:
    """Generate tokens using top-k sampling."""
    return generate(model, prompt, max_len, TopKSampling(k, temperature, seed))


def generate_top_p(
    model,
    prompt: list[int],
    max_len: int,
    top_p: float = 0.9,
    temperature: float = 1.0,
    seed: int = 42,
) -> list[int]:
    """Generate tokens using top-p (nucleus) sampling."""
    return generate(model, prompt, max_len, TopPSampling(top_p, temperature, seed))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _next_rand01(state: list[int]) -> float:
    """LCG PRNG returning [0, 1).

    Uses a Linear Congruential Generator with the same constants as
    PCG's default multiplier.  State is a 1-element list for mutability
    (Python ints are immutable, so we mutate the list in-place).
    This provides cross-platform deterministic randomness independent
    of numpy's global RNG state.
    """
    state[0] = (state[0] * 6_364_136_223_846_793_005 + 1) & 0xFFFFFFFFFFFFFFFF
    return float((state[0] >> 32) & 0xFFFFFFFF) / 4294967296.0


def _softmax(xs: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D array.

    Softmax: p_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    """
    shifted = xs - np.max(xs)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if total == 0:
        return exp_vals
    return exp_vals / total


def _sample_from_probs(probs: np.ndarray, state: list[int]) -> int:
    """Sample an index from a probability distribution via inverse CDF.

    Draws a uniform random number and walks the cumulative distribution
    until the threshold is crossed.
    """
    r = _next_rand01(state)
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def _sample_from_logits(
    logits: np.ndarray, temperature: float, state: list[int]
) -> int:
    """Sample from logits with temperature scaling.

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    token = sample(probs)

    Temperature=0 degenerates to greedy (argmax).
    """
    if temperature <= 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = _softmax(scaled)
    return _sample_from_probs(probs, state)


def _sample_top_k(
    logits: np.ndarray, k: int, temperature: float, state: list[int]
) -> int:
    """Sample from top-k filtered logits.

    Masks all tokens outside the top-k to -inf before applying
    temperature scaling and softmax.
    """
    n = len(logits)
    if k <= 0 or k >= n:
        return _sample_from_logits(logits, temperature, state)

    indices = np.argsort(logits)[::-1]
    filtered = np.full(n, -np.inf)
    for i in range(k):
        filtered[indices[i]] = logits[indices[i]]
    return _sample_from_logits(filtered, temperature, state)


def _sample_top_p(
    logits: np.ndarray, top_p: float, temperature: float, state: list[int]
) -> int:
    """Sample from top-p (nucleus) filtered logits.

    Sorts tokens by descending probability, includes tokens until the
    cumulative probability exceeds top_p, then renormalizes and samples.
    """
    if top_p <= 0 or top_p >= 1:
        return _sample_from_logits(logits, temperature, state)
    if temperature <= 0:
        return int(np.argmax(logits))

    scaled = logits / temperature
    probs = _softmax(scaled)

    # Sort by descending probability and accumulate until threshold
    indices = np.argsort(probs)[::-1]
    cum = 0.0
    selected = []
    for idx in indices:
        selected.append(idx)
        cum += probs[idx]
        if cum >= top_p:
            break

    # Renormalize the selected subset
    trunc = np.array([probs[idx] for idx in selected])
    total = np.sum(trunc)
    if total > 0:
        trunc /= total

    chosen = _sample_from_probs(trunc, state)
    return int(selected[chosen])
