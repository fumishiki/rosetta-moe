use ffi_julia::{JuliaEngine, Result};

fn main() -> Result<()> {
    println!("Initializing Julia runtime...");
    let mut engine = JuliaEngine::new()?;

    println!("Loading MLCore module...");
    let module_path = std::env::current_dir()?
        .join("../../julia/src/MLCore.jl")
        .canonicalize()?;
    engine.load_module(module_path.to_str().unwrap())?;

    println!("Creating test data...");
    let batch = 2;
    let seq = 4;
    let ids: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let targets: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    println!("Running forward pass...");
    let data = vec![0.1f32; 2 * 4 * 256];
    let result = engine.forward(&data, [2, 4, 256])?;
    println!("Output shape: {}", result.len());

    println!("Running training step...");
    let loss = engine.train_step(&ids, &targets, [batch, seq])?;
    println!("Loss: {:.4}", loss);

    println!("Success!");
    Ok(())
}
