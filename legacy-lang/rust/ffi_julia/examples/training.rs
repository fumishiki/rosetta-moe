use ffi_julia::{JuliaEngineBuilder, Result};

fn main() -> Result<()> {
    // Initialize with dynamic patching enabled
    let mut engine = JuliaEngineBuilder::new()
        .config("../julia/config.yaml")
        .model("1.9B")
        .module("../julia/src/MLCore.jl")
        .with_patching(true) // Enable entropy-based dynamic patching
        .build()?;

    println!("Julia engine initialized with dynamic patching!");
    println!("Model config: {:?}", engine.model_config());
    println!("Training config: {:?}", engine.training_config());
    println!("Patching config: {:?}", engine.patching_config());

    // Training data (byte sequences)
    let train_text = "The quick brown fox jumps over the lazy dog";
    let train_bytes: Vec<u8> = train_text.bytes().collect();

    // Prepare input and target (shifted by 1)
    let seq_len = train_bytes.len() - 1;
    let inputs = &train_bytes[..seq_len];
    let targets = &train_bytes[1..];

    println!("\nTraining on: {}", train_text);
    println!("Sequence length: {}", seq_len);

    // Get learning rate from config
    let lr = engine
        .training_config()
        .map(|cfg| cfg.learning_rate)
        .unwrap_or(1e-3);

    // Training loop
    let epochs = 10;
    for epoch in 1..=epochs {
        let loss = engine.train_step(
            inputs,
            targets,
            [1, seq_len],
            lr,
            false, // CPU training
        )?;

        println!("Epoch {}/{}: Loss = {:.4}", epoch, epochs, loss);
    }

    // Test generation after training
    println!("\nGenerating after training...");
    let prompt = "The quick".bytes().collect::<Vec<_>>();
    let generated = engine.generate(&prompt, 50, 1.0, false)?;
    let generated_text = String::from_utf8_lossy(&generated);
    println!("Generated: {}", generated_text);

    Ok(())
}
