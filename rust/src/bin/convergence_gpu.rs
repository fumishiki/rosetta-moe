// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! GPU loss convergence verification.
//! Trains the tiny MoE Transformer for 500 steps using Metal GPU acceleration.

use nn_core::MoETransformer;

#[cfg(feature = "metal")]
use nn_core::{GpuModel, MetalContext, MetalTensor, gpu_train_step};

fn main() {
    #[cfg(not(feature = "metal"))]
    {
        eprintln!("Metal feature not enabled");
        std::process::exit(1);
    }

    #[cfg(feature = "metal")]
    {
        nn_core::seed_rng(42);
        let model = MoETransformer::tiny();

        let mut ctx = MetalContext::new().expect("Metal not available");
        ctx.load_required_shaders().expect("Failed to load shaders");

        let gpu_model = GpuModel::from_cpu(&ctx, &model).expect("Failed to upload model to GPU");

        let batch = 2;
        let seq = 8;
        let vocab = 1000;

        // Fixed deterministic input (same across all 4 languages)
        let input_data: Vec<f32> = (0..batch * seq).map(|i| (i % vocab) as f32).collect();
        let input = MetalTensor::upload(&ctx, &input_data, vec![batch, seq])
            .expect("Failed to create input tensor");

        let target_data: Vec<f32> = (0..batch * seq).map(|i| ((i + 1) % vocab) as f32).collect();
        let targets = MetalTensor::upload(&ctx, &target_data, vec![batch, seq])
            .expect("Failed to create target tensor");

        let n_steps = 500;
        let mut losses: Vec<f32> = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let loss = gpu_train_step(&ctx, &gpu_model, &input, &targets)
                .expect("GPU training step failed");
            losses.push(loss);
        }

        // Output JSON
        let losses_str: Vec<String> = losses.iter().map(|l| format!("{l:.6}")).collect();
        println!(
            "{{\"language\":\"rust\",\"steps\":{n_steps},\"losses\":[{}]}}",
            losses_str.join(",")
        );
    }
}
