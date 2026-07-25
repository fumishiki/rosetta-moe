use ffi_julia::{JuliaEngine, Result};

#[test]
fn test_julia_runtime_init() -> Result<()> {
    let _engine = JuliaEngine::new()?;
    Ok(())
}

#[test]
fn test_module_loading() -> Result<()> {
    let mut engine = JuliaEngine::new()?;
    let module_path = std::env::current_dir()?
        .join("../../julia/src/MLCore.jl")
        .canonicalize()?;
    engine.load_module(module_path.to_str().unwrap())?;
    Ok(())
}

#[test]
fn test_zero_copy_array() -> Result<()> {
    let mut engine = JuliaEngine::new()?;
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    engine.with(|frame| {
        use jlrs::prelude::*;
        unsafe {
            let arr = TypedArray::from_slice(frame, &data, [2, 2])?;
            let slice = arr.inline_data()?;
            assert_eq!(slice.len(), 4);
            assert_eq!(slice[0], 1.0);
        }
        Ok(())
    })
}

#[test]
fn test_forward_pass() -> Result<()> {
    let mut engine = JuliaEngine::new()?;
    let module_path = std::env::current_dir()?
        .join("../../julia/src/MLCore.jl")
        .canonicalize()?;
    engine.load_module(module_path.to_str().unwrap())?;

    let data = vec![0.1f32; 1 * 4 * 256];
    let result = engine.forward(&data, [1, 4, 256])?;
    assert_eq!(result.len(), 1 * 4 * 256);
    Ok(())
}
