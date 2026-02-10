// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Loss convergence verification.
//! Trains the tiny MoE Transformer for 200 steps and outputs loss per step.

use nn_core::{MoETransformer, Shape, Tensor, TrainConfig, Trainer};

fn main() {
    let model = MoETransformer::tiny();
    let train_cfg = TrainConfig {
        batch_size: 2,
        seq_len: 8,
        lr: 1e-3,
        warmup_steps: 10,
        total_steps: 1200,
        grad_clip: 1.0,
        aux_loss_weight: 0.01,
    };
    let mut trainer = Trainer::new(model, train_cfg);

    let batch = 2;
    let seq = 8;
    let vocab = 1000;

    // Fixed deterministic input (same across all 4 languages)
    let input_data: Vec<f32> = (0..batch * seq).map(|i| (i % vocab) as f32).collect();
    let input = Tensor::from_slice(&input_data, Shape::new(&[batch, seq]));

    let target_data: Vec<f32> = (0..batch * seq).map(|i| ((i + 1) % vocab) as f32).collect();
    let targets = Tensor::from_slice(&target_data, Shape::new(&[batch, seq]));

    let n_steps = 1000;
    let mut losses: Vec<f32> = Vec::with_capacity(n_steps);

    for _ in 0..n_steps {
        let loss = trainer.train_step(&input, &targets);
        losses.push(loss);
    }

    // Output JSON
    let losses_str: Vec<String> = losses.iter().map(|l| format!("{l:.6}")).collect();
    println!(
        "{{\"language\":\"rust\",\"steps\":{n_steps},\"losses\":[{}]}}",
        losses_str.join(",")
    );
}
