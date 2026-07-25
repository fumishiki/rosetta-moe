// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

// RoPE in-place: rotates pairs [x0,x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
// x: [batch_seq, n_heads, head_dim] flattened
// cos_cache/sin_cache: [max_seq, half_dim] precomputed
// Each thread handles one (token, head, pair) tuple
kernel void rope_inplace(
    device float* x [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_pairs_per_token = n_heads * half_dim;
    uint token_idx = id / total_pairs_per_token;
    uint remainder = id % total_pairs_per_token;
    uint head_idx = remainder / half_dim;
    uint pair_idx = remainder % half_dim;

    // Position in sequence (within batch)
    uint seq_pos = token_idx % seq_len;

    // Offset into x: token_idx * (n_heads * head_dim) + head_idx * head_dim + pair_idx * 2
    uint base = token_idx * n_heads * head_dim + head_idx * head_dim;
    uint i0 = base + pair_idx * 2;
    uint i1 = i0 + 1;

    // Frequency index
    uint freq_idx = seq_pos * half_dim + pair_idx;
    float cos_val = cos_cache[freq_idx];
    float sin_val = sin_cache[freq_idx];

    float x0 = x[i0];
    float x1 = x[i1];
    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

// Causal mask fill: set scores[qi, ki] = -inf for ki > qi
// scores: [batch * n_heads, seq_len, seq_len] flattened, stored row-major
// Each thread handles one element in the scores matrix
kernel void causal_mask_fill(
    device float* scores [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    uint total_per_head = seq_len * seq_len;
    uint within_head = id % total_per_head;
    uint qi = within_head / seq_len;
    uint ki = within_head % seq_len;
    if (ki > qi) {
        scores[id] = -INFINITY;
    }
}

// Attention scores kernel:
// q:      [batch_seq, n_heads * head_dim]
// k:      [batch_seq, n_kv_heads * head_dim]
// scores: [batch * n_heads, seq_len, seq_len]
kernel void attention_scores(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    uint elems_per_head = seq_len * seq_len;
    uint bh = id / elems_per_head;
    uint within = id % elems_per_head;
    uint q_pos = within / seq_len;
    uint k_pos = within % seq_len;

    uint batch_idx = bh / n_heads;
    uint head_idx = bh % n_heads;
    uint kv_head = head_idx % n_kv_heads;

    uint q_token = batch_idx * seq_len + q_pos;
    uint k_token = batch_idx * seq_len + k_pos;

    uint q_base = q_token * n_heads * head_dim + head_idx * head_dim;
    uint k_base = k_token * n_kv_heads * head_dim + kv_head * head_dim;

    float acc = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        acc += q[q_base + d] * k[k_base + d];
    }
    scores[id] = acc * scale;
}

// Attention weighted sum kernel:
// weights: [batch * n_heads, seq_len, seq_len]
// v:       [batch_seq, n_kv_heads * head_dim]
// out:     [batch_seq, n_heads * head_dim]
kernel void attention_weighted_sum(
    device const float* weights [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_kv_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint head_span = n_heads * head_dim;
    uint token = id / head_span;
    uint rem = id % head_span;
    uint head_idx = rem / head_dim;
    uint dim_idx = rem % head_dim;

    uint batch_idx = token / seq_len;
    uint q_pos = token % seq_len;
    uint kv_head = head_idx % n_kv_heads;

    uint weight_base = (batch_idx * n_heads + head_idx) * seq_len * seq_len + q_pos * seq_len;

    float acc = 0.0f;
    for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
        uint src_token = batch_idx * seq_len + k_pos;
        uint v_idx = src_token * n_kv_heads * head_dim + kv_head * head_dim + dim_idx;
        acc += weights[weight_base + k_pos] * v[v_idx];
    }
    out[id] = acc;
}

// MoE router: gate_probs = softmax(input @ gate_weight^T)
// Handled by gpu_linear + softmax kernel.

// MoE top-k selection: per-token, find top-k expert indices and weights
// token_probs: [batch_seq, n_experts] — post-softmax gate probs
// out_indices: [batch_seq, top_k] — selected expert indices (as float for buffer compat)
// out_weights: [batch_seq, top_k] — renormalized weights
// Each threadgroup handles one token
kernel void moe_topk(
    device const float* token_probs [[buffer(0)]],
    device float* out_indices [[buffer(1)]],
    device float* out_weights [[buffer(2)]],
    constant uint& n_experts [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]]
) {
    uint row_offset = gid * n_experts;
    uint out_offset = gid * top_k;

    // Simple greedy top-k (n_experts is small, typically 4-16)
    // Use local array for selection tracking
    // Note: top_k is typically 2-4, n_experts typically 4-16
    float selected_probs[16]; // max experts
    uint selected_idx[16];
    bool used[16];

    for (uint e = 0; e < n_experts && e < 16; e++) {
        used[e] = false;
    }

    for (uint k = 0; k < top_k; k++) {
        float best_val = -INFINITY;
        uint best_idx = 0;
        for (uint e = 0; e < n_experts && e < 16; e++) {
            if (!used[e] && token_probs[row_offset + e] > best_val) {
                best_val = token_probs[row_offset + e];
                best_idx = e;
            }
        }
        selected_probs[k] = best_val;
        selected_idx[k] = best_idx;
        used[best_idx] = true;
    }

    // Renormalize weights
    float sum = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        sum += selected_probs[k];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    for (uint k = 0; k < top_k; k++) {
        out_indices[out_offset + k] = float(selected_idx[k]);
        out_weights[out_offset + k] = selected_probs[k] * inv_sum;
    }
}

// Extract one expert's per-token routing weights from top-k results.
// indices: [batch_seq, top_k] (float-encoded expert indices)
// weights: [batch_seq, top_k]
// out: [batch_seq]
kernel void moe_topk_extract_weight(
    device const float* indices [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& top_k [[buffer(3)]],
    constant uint& expert_idx [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id;
    uint base = row * top_k;
    float sum = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        uint e = uint(indices[base + k]);
        if (e == expert_idx) {
            sum += weights[base + k];
        }
    }
    out[row] = sum;
}

// Row-wise scaling: out[row, col] = input[row, col] * row_weights[row]
kernel void row_scale(
    device const float* input [[buffer(0)]],
    device const float* row_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_cols [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id / n_cols;
    output[id] = input[id] * row_weights[row];
}

// MoE weighted scatter-add: output[token] += weight * expert_output[token]
// For each token assigned to this expert, add weighted expert output
// expert_out: [n_assigned, hidden] — output of one expert for assigned tokens
// token_map: [n_assigned] — which global token index each assigned token corresponds to
// weights: [n_assigned] — expert weight for each assigned token
// output: [batch_seq, hidden] — accumulated output (must be pre-zeroed)
kernel void moe_scatter_add(
    device const float* expert_out [[buffer(0)]],
    device const float* token_map [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint assigned_idx = id / hidden_dim;
    uint dim_idx = id % hidden_dim;
    uint global_token = uint(token_map[assigned_idx]);
    float w = weights[assigned_idx];
    // Atomic add not needed if we process experts sequentially
    output[global_token * hidden_dim + dim_idx] += w * expert_out[assigned_idx * hidden_dim + dim_idx];
}

// Gather tokens for an expert: output[i] = input[token_map[i]]
// input: [batch_seq, hidden], token_map: [n_assigned] (float-encoded indices), output: [n_assigned, hidden]
kernel void moe_gather(
    device const float* input [[buffer(0)]],
    device const float* token_map [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint assigned_idx = id / hidden_dim;
    uint dim_idx = id % hidden_dim;
    uint src_token = uint(token_map[assigned_idx]);
    output[id] = input[src_token * hidden_dim + dim_idx];
}

// Transpose weight: out[c * rows + r] = in[r * cols + c]
// For transposing [out_features, in_features] -> [in_features, out_features]
kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint r = id / cols;
    uint c = id % cols;
    output[c * rows + r] = input[id];
}

// Cross-entropy loss forward: loss = -sum(log(softmax(logits)[target])) / n_tokens
// logits: [batch_seq, vocab], targets: [batch_seq] (float-encoded indices)
// output: [batch_seq] — per-token loss
kernel void cross_entropy_forward(
    device const float* logits [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_data[256];

    uint row_offset = gid * vocab_size;
    uint target_idx = uint(targets[gid]);

    // Find max for numerical stability
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        local_max = max(local_max, logits[row_offset + i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        local_sum += exp(logits[row_offset + i] - row_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_data[0];

    // loss = -(logits[target] - max - log(sum))
    if (tid == 0) {
        float log_softmax = logits[row_offset + target_idx] - row_max - log(total_sum);
        output[gid] = -log_softmax;
    }
}

// Cross-entropy backward: grad_logits = softmax(logits) - one_hot(target)
// logits: [batch_seq, vocab], targets: [batch_seq], grad_out: [batch_seq, vocab]
kernel void cross_entropy_backward(
    device const float* logits [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device float* grad_out [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_data[256];

    uint row_offset = gid * vocab_size;
    uint target_idx = uint(targets[gid]);

    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        local_max = max(local_max, logits[row_offset + i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sum of exp
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab_size; i += tg_size) {
        local_sum += exp(logits[row_offset + i] - row_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // grad = scale * (softmax(logits) - one_hot(target))
    for (uint i = tid; i < vocab_size; i += tg_size) {
        float softmax_val = exp(logits[row_offset + i] - row_max) / total_sum;
        float target_val = (i == target_idx) ? 1.0f : 0.0f;
        grad_out[row_offset + i] = scale * (softmax_val - target_val);
    }
}

// Reduce sum: computes sum of input array, writes to output[0]
// For aggregating per-token losses into a scalar
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];

    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        local_sum += input[i];
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_data[0];
    }
}
