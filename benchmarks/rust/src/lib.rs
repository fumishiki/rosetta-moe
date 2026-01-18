//! Benchmark utilities for cross-language comparison

/// Matrix multiplication (naive)
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Softmax (row-wise)
pub fn softmax(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    for r in 0..rows {
        let offset = r * cols;
        let row = &input[offset..offset + cols];

        // Find max for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp and sum
        let mut sum = 0.0f32;
        for c in 0..cols {
            let exp_val = (row[c] - max_val).exp();
            output[offset + c] = exp_val;
            sum += exp_val;
        }

        // Normalize
        for c in 0..cols {
            output[offset + c] /= sum;
        }
    }
    output
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect()
}

/// RMSNorm
pub fn rmsnorm(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let n = input.len() / dim;
    let mut output = vec![0.0f32; input.len()];

    for i in 0..n {
        let offset = i * dim;
        let slice = &input[offset..offset + dim];

        // Compute RMS
        let sum_sq: f32 = slice.iter().map(|x| x * x).sum();
        let rms = (sum_sq / dim as f32 + eps).sqrt();

        // Normalize and scale
        for j in 0..dim {
            output[offset + j] = (slice[j] / rms) * weight[j];
        }
    }
    output
}

/// Generate random f32 vector
pub fn random_vec(size: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..size)
        .map(|_| {
            // Simple LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}
