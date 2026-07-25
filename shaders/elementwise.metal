// SPDX-License-Identifier: CC-BY-NC-SA-4.0
#include <metal_stdlib>
using namespace metal;

// Vector addition: out = a + b
kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] + b[id];
}

// Vector scale: out = a * scalar
kernel void scale(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * scalar;
}

// Elementwise multiply: out = a * b
kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = a[id] * b[id];
}

// Fused add + scale: out = (a + b) * scalar
kernel void add_scale(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scalar [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    out[id] = (a[id] + b[id]) * scalar;
}
