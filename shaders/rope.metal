// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

// RoPE rotation for a single head
// input layout: [batch, seq, n_heads, head_dim] flattened
// freqs: [seq, head_dim/2] — precomputed cos/sin pairs
kernel void rope(
    device float* x [[buffer(0)]],
    device const float* cos_freqs [[buffer(1)]],
    device const float* sin_freqs [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // Each thread handles one pair (2 elements)
    uint pair_idx = id;
    uint half_dim = head_dim / 2;
    uint pos_in_head = pair_idx % half_dim;
    uint head_and_above = pair_idx / half_dim;

    uint base = head_and_above * head_dim;
    uint i0 = base + pos_in_head;
    uint i1 = base + pos_in_head + half_dim;

    // Frequency index: position in sequence * half_dim + pair position
    uint seq_pos = (head_and_above / 1) % 512; // simplified — actual seq pos from params
    uint freq_idx = seq_pos * half_dim + pos_in_head;

    float cos_val = cos_freqs[freq_idx];
    float sin_val = sin_freqs[freq_idx];

    float x0 = x[i0];
    float x1 = x[i1];

    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}
