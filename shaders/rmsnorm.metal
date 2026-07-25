// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_data[256];

    uint row_offset = gid * hidden_dim;

    // Sum of squares (parallel reduction)
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        float val = input[row_offset + i];
        local_sum_sq += val * val;
    }
    shared_data[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_data[0] / float(hidden_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale by weight
    for (uint i = tid; i < hidden_dim; i += tg_size) {
        output[row_offset + i] = input[row_offset + i] * rms * weight[i];
    }
}
