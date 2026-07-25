use ffi_julia::{JuliaEngineBuilder, Result};

fn main() -> Result<()> {
    // Initialize Julia engine with config
    let mut engine = JuliaEngineBuilder::new()
        .config("../julia/config.yaml")
        .model("1.9B")
        .module("../julia/src/MLCore.jl")
        .with_patching(false)
        .build()?;

    println!("Julia engine initialized!");
    println!("Model config: {:?}", engine.model_config());

    // Prepare input (byte sequence)
    let input_text = "Hello, world!";
    let input_bytes: Vec<u8> = input_text.bytes().collect();
    let batch_size = 1;
    let seq_len = input_bytes.len();

    println!("\nInput: {}", input_text);
    println!("Input bytes: {:?}", input_bytes);

    // Forward pass (CPU)
    println!("\nRunning forward pass on CPU...");
    let logits = engine.forward(&input_bytes, [batch_size, seq_len], false)?;
    println!("Output logits shape: {} elements", logits.len());
    println!("First 10 logits: {:?}", &logits[..10.min(logits.len())]);

    // Generate text
    println!("\nGenerating text...");
    let prompt = "Hello".bytes().collect::<Vec<_>>();
    let generated = engine.generate(&prompt, 100, 1.0, false)?;
    let generated_text = String::from_utf8_lossy(&generated);
    println!("Generated: {}", generated_text);

    Ok(())
}
