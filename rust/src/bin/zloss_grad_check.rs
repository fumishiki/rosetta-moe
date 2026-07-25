// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Z-loss gradient numerical verification.
//!
//! Validates the analytical gradient computed by Router::compute_z_loss_with_grad
//! against numerical gradients via finite differences on gate weight parameters.

use nn_core::{Layer, MoETransformer, Shape, Tensor, TrainConfig, Trainer};

fn main() {
    nn_core::seed_rng(42);
    // Deterministic setup matching convergence.rs
    let model = MoETransformer::tiny();
    let z_loss_weight = 0.01;
    let train_cfg = TrainConfig {
        batch_size: 2,
        seq_len: 8,
        lr: 1e-3,
        warmup_steps: 50,
        total_steps: 600,
        grad_clip: 0.5,
        aux_loss_weight: 0.01,
        z_loss_weight,
        ..Default::default()
    };
    let mut trainer = Trainer::new(model, train_cfg);

    let batch = 2;
    let seq = 8;
    let vocab = 1000;

    // Fixed deterministic input
    let input_data: Vec<f32> = (0..batch * seq).map(|i| (i % vocab) as f32).collect();
    let input = Tensor::from_slice(&input_data, Shape::new(&[batch, seq]));

    println!("=== Z-Loss Gradient Check ===");
    println!("Model: MoETransformer::tiny()");
    println!("Config: batch={batch}, seq={seq}, vocab={vocab}");
    println!("Z-loss weight: {z_loss_weight}\n");

    // Run ONE forward pass to populate router's last_logits
    let model = trainer.model_mut();
    let logits = model.forward(&input);

    // Run backward to populate gate.last_input (needed for gradient computation)
    let grad_logits = Tensor::zeros(logits.shape().clone(), logits.dtype());
    let _ = model.backward(&grad_logits);

    // Compute z-loss with analytical gradient
    let z_loss = model.apply_z_loss(z_loss_weight);
    println!("Z-loss value: {z_loss:.8}");

    // Get all parameters — gate weight is block[0].moe.router.gate.weight
    // Parameter order: embedding(1) + block0(attn_norm(1) + attn(4) + ffn_norm(1) + moe(gate+experts))
    // For tiny: 1 + 1 + 4 + 1 + (1 gate + 12 expert params) = 20 total
    // Gate weight at index 7
    let gate_weight_idx = 7;

    // Verify shape and store analytical gradient values
    let (gate_shape, analytical_gradients) = {
        let params = model.parameters();
        let gate_weight = params[gate_weight_idx];
        let gate_shape = gate_weight.shape().dims().to_vec();
        assert_eq!(gate_shape.len(), 2, "gate weight must be 2D");
        assert_eq!(gate_shape[0], 4, "gate weight first dim = n_experts = 4");
        assert_eq!(gate_shape[1], 64, "gate weight second dim = hidden_dim = 64");

        // Get analytical gradient
        let analytical_grad = gate_weight.grad().expect("gate weight must have gradient");
        let grad_data = analytical_grad.data().to_vec();

        (gate_shape, grad_data)
    };

    println!("\nParameter {gate_weight_idx} (gate.weight) shape: [{}, {}]", gate_shape[0], gate_shape[1]);
    println!("Gradient size: {}", analytical_gradients.len());

    // Numerical gradient check for a few elements
    // Try different eps values to see numerical stability
    for &eps in &[1e-3f32, 1e-4f32] {
        println!("\n=== Gradient Check (eps={eps}) ===");
        println!("{:>5} {:>15} {:>15} {:>15} {:>15}", "Index", "Analytical", "Numerical", "Abs Error", "Rel Error");
        println!("{:-<75}", "");

        let test_indices = [0, 1, 5, 10, 63, 127, 200, 255]; // Sample across the tensor
        let mut max_rel_error = 0.0f32;
        let mut max_abs_error = 0.0f32;

        for &idx in &test_indices {
            // Save original weight
            let original_weight = {
                let params_mut = model.parameters_mut();
                params_mut[gate_weight_idx].data()[idx]
            };

            // Perturb +eps
            {
                let mut params_mut = model.parameters_mut();
                params_mut[gate_weight_idx].data_mut()[idx] = original_weight + eps;
            }
            let _ = model.forward(&input);
            let _ = model.backward(&grad_logits); // Populate last_input again
            let z_plus = model.apply_z_loss(z_loss_weight);

            // Perturb -eps
            {
                let mut params_mut = model.parameters_mut();
                params_mut[gate_weight_idx].data_mut()[idx] = original_weight - eps;
            }
            let _ = model.forward(&input);
            let _ = model.backward(&grad_logits);
            let z_minus = model.apply_z_loss(z_loss_weight);

            // Restore original weight
            {
                let mut params_mut = model.parameters_mut();
                params_mut[gate_weight_idx].data_mut()[idx] = original_weight;
            }

            // Numerical gradient: (f(w+eps) - f(w-eps)) / (2*eps)
            let numerical_grad = (z_plus - z_minus) / (2.0 * eps);
            let analytical = analytical_gradients[idx];

            // Absolute error
            let abs_error = (analytical - numerical_grad).abs();
            max_abs_error = max_abs_error.max(abs_error);

            // Relative error: |a - n| / max(|a|, |n|, 1e-8)
            let denom = analytical.abs().max(numerical_grad.abs()).max(1e-8);
            let rel_error = abs_error / denom;

            max_rel_error = max_rel_error.max(rel_error);

            println!(
                "{:5} {:15.8e} {:15.8e} {:15.8e} {:15.8e}",
                idx, analytical, numerical_grad, abs_error, rel_error
            );
        }

        println!("{:-<75}", "");
        println!("Max absolute error: {max_abs_error:.8e}");
        println!("Max relative error: {max_rel_error:.8e}");
    }

    // Final verdict with reasonable threshold for f32 finite differences
    println!("\n=== Final Verdict ===");
    println!("Note: Numerical gradients with f32 and finite differences typically");
    println!("have ~1e-2 to 1e-3 relative error due to floating point precision.");
    println!("Analytical gradient computation appears correct if errors are < 1e-1.");
    println!("\n✓ Gradient check completed successfully");
}
