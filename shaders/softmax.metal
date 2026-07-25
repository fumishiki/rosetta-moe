// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    threadgroup float shared_data[256];

    // Row offset
    uint row_offset = gid * n;

    // Pass 1: find max (parallel reduction)
    float local_max = -INFINITY;
    for (uint i = tid; i < n; i += tg_size) {
        local_max = max(local_max, input[row_offset + i]);
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

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = val;
        local_sum += val;
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

    // Pass 3: normalize
    for (uint i = tid; i < n; i += tg_size) {
        output[row_offset + i] /= total_sum;
    }
}
