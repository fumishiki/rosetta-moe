// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}
