// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Loss convergence verification.
//! Trains the tiny MoE Transformer for 500 steps and outputs loss per step.

use nn_core::{MoETransformer, RoutingMode, Shape, Tensor, TrainConfig, Trainer};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let routing_mode = if let Some(idx) = args.iter().position(|a| a == "--routing-mode") {
        if let Some(mode_str) = args.get(idx + 1) {
            match mode_str.to_lowercase().as_str() {
                "topk" => RoutingMode::TopK,
                "biasfree" => RoutingMode::BiasFree,
                "relu" => RoutingMode::ReLU,
                _ => {
                    eprintln!("Invalid routing mode: {mode_str}. Using default TopK.");
                    RoutingMode::TopK
                }
            }
        } else {
            RoutingMode::TopK
        }
    } else {
        RoutingMode::TopK
    };

    let seed = if let Some(idx) = args.iter().position(|a| a == "--seed") {
        args.get(idx + 1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(42)
    } else {
        42
    };
    nn_core::seed_rng(seed);
    let mut model = MoETransformer::tiny();
    model.set_routing_mode(routing_mode);

    let train_cfg = TrainConfig {
        batch_size: 2,
        seq_len: 8,
        lr: 1e-3,
        warmup_steps: 50,
        total_steps: 600,
        grad_clip: 0.5,
        aux_loss_weight: 0.01,
        z_loss_weight: 0.05,
        routing_mode,
        bias_gamma: 0.001,
        relu_lambda_l1: 0.01,
        relu_target_k: 2,
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

    let diag = args.iter().any(|a| a == "--diag");
    let n_steps = 500;
    let mut losses: Vec<f32> = Vec::with_capacity(n_steps);

    for step in 1..=n_steps {
        let loss = trainer.train_step(&input, &targets);
        losses.push(loss);
        if diag && (step <= 5 || step % 50 == 0 || step % 10 == 0 && step <= 100) {
            let (ce, aux, z, gnorm, clipped) = trainer.last_loss_components();
            let clip_flag = if clipped { " CLIP" } else { "" };
            let expert_strs: Vec<String> = trainer.expert_counts().iter()
                .enumerate()
                .map(|(layer, counts)| {
                    let cs: Vec<String> = counts.iter().map(|c| format!("{c:.0}")).collect();
                    format!("L{layer}=[{}]", cs.join(","))
                })
                .collect();
            eprintln!("step {step:>3}: total={loss:.6} ce={ce:.6} aux={aux:.6} z={z:.6} gnorm={gnorm:.4}{clip_flag} experts:{}", expert_strs.join(" "));
        }
    }

    // Output JSON
    let losses_str: Vec<String> = losses.iter().map(|l| format!("{l:.6}")).collect();
    println!(
        "{{\"language\":\"rust\",\"steps\":{n_steps},\"losses\":[{}]}}",
        losses_str.join(",")
    );
}
