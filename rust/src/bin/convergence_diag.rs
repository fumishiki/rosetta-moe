// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Loss convergence diagnostic: z_loss_weight regression analysis.
//! Runs two training scenarios to compare convergence:
//! - Run 1: z_loss_weight = 0.0
//! - Run 2: z_loss_weight = 0.01 (production)

use nn_core::{MoETransformer, Shape, Tensor, TrainConfig, Trainer};

fn main() {
    let batch = 2;
    let seq = 8;
    let vocab = 1000;

    // Fixed deterministic input (same across all 4 languages)
    let input_data: Vec<f32> = (0..batch * seq).map(|i| (i % vocab) as f32).collect();
    let input = Tensor::from_slice(&input_data, Shape::new(&[batch, seq]));

    let target_data: Vec<f32> = (0..batch * seq).map(|i| ((i + 1) % vocab) as f32).collect();
    let targets = Tensor::from_slice(&target_data, Shape::new(&[batch, seq]));

    let n_steps = 600;

    // Run 1: z_loss_weight = 0.0
    eprintln!("=== z_loss_weight=0.00 ===");
    nn_core::seed_rng(42);
    let model = MoETransformer::tiny();
    let train_cfg = TrainConfig {
        batch_size: 2,
        seq_len: 8,
        lr: 1e-3,
        warmup_steps: 50,
        total_steps: 600,
        grad_clip: 0.5,
        aux_loss_weight: 0.01,
        z_loss_weight: 0.0,
        ..Default::default()
    };
    let mut trainer = Trainer::new(model, train_cfg);

    let mut losses_run1: Vec<f32> = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let loss = trainer.train_step(&input, &targets);
        losses_run1.push(loss);
    }

    let final_loss_1 = losses_run1.last().copied().unwrap_or(f32::NAN);
    let (min_loss_1, min_step_1) = losses_run1
        .iter()
        .enumerate()
        .fold((f32::INFINITY, 0), |(min_l, min_s), (step, &l)| {
            if l < min_l {
                (l, step)
            } else {
                (min_l, min_s)
            }
        });

    eprintln!("Final loss: {:.6}", final_loss_1);
    eprintln!("Min loss: {:.6} at step {}", min_loss_1, min_step_1);
    eprintln!();

    // Run 2: z_loss_weight = 0.01
    eprintln!("=== z_loss_weight=0.01 ===");
    nn_core::seed_rng(42);
    let model = MoETransformer::tiny();
    let train_cfg = TrainConfig {
        batch_size: 2,
        seq_len: 8,
        lr: 1e-3,
        warmup_steps: 50,
        total_steps: 600,
        grad_clip: 0.5,
        aux_loss_weight: 0.01,
        z_loss_weight: 0.01,
        ..Default::default()
    };
    let mut trainer = Trainer::new(model, train_cfg);

    let mut losses_run2: Vec<f32> = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let loss = trainer.train_step(&input, &targets);
        losses_run2.push(loss);
    }

    let final_loss_2 = losses_run2.last().copied().unwrap_or(f32::NAN);
    let (min_loss_2, min_step_2) = losses_run2
        .iter()
        .enumerate()
        .fold((f32::INFINITY, 0), |(min_l, min_s), (step, &l)| {
            if l < min_l {
                (l, step)
            } else {
                (min_l, min_s)
            }
        });

    eprintln!("Final loss: {:.6}", final_loss_2);
    eprintln!("Min loss: {:.6} at step {}", min_loss_2, min_step_2);
}
