// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

// Embedding gather: output[i] = weight[token_ids[i]]
// token_ids: [batch_seq] float (cast to uint), weight: [vocab, hidden], output: [batch_seq, hidden]
kernel void embedding_gather(
    device const float* token_ids [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint token_idx = id / hidden_dim;
    uint dim_idx = id % hidden_dim;
    uint tid = uint(token_ids[token_idx]);
    output[id] = weight[tid * hidden_dim + dim_idx];
}
