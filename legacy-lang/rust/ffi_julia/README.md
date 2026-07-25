# ffi_julia - Rust ⟷ Julia FFI for MLCore

Type-safe Rust bindings for the Julia MLCore.jl BLT Recurrent MoE Transformer.

## Features

- **Config-driven**: Load model settings from `config.yaml`
- **Type-safe API**: Rust structs mirror Julia types
- **Dynamic Patching**: Enable entropy-based byte patching
- **GPU Support**: CPU/GPU training and inference
- **Builder Pattern**: Ergonomic model initialization
- **Zero-copy**: Efficient data transfer via jlrs

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
ffi_julia = { path = "path/to/ffi_julia" }
```

## Quick Start

Initialize the engine with `JuliaEngineBuilder`, load config from YAML, and run inference/training/generation.

- **Basic Inference**: Forward pass on byte sequences
- **Dynamic Patching**: Enable entropy-based patching with `.with_patching(true)` for 50% FLOP reduction
- **Training**: Run training steps with custom learning rate and GPU support
- **Text Generation**: Generate sequences with temperature-controlled sampling

See [examples/inference.rs](examples/inference.rs) and [examples/training.rs](examples/training.rs) for complete implementations.

## API Reference

### JuliaEngine

Main interface to Julia runtime.

```rust
impl JuliaEngine {
    fn new() -> Result<Self>
    fn with_config(config: FullConfig, model_name: impl Into<String>) -> Result<Self>
    fn load_config(&mut self, path: impl AsRef<Path>, model_name: impl Into<String>) -> Result<&Self>
    fn load_module(&mut self, path: &str) -> Result<()>
    fn create_model(&mut self, use_patching: bool) -> Result<()>

    fn forward(&mut self, byte_ids: &[u8], dims: [usize; 2], gpu: bool) -> Result<Vec<f32>>
    fn train_step(&mut self, ids: &[u8], targets: &[u8], dims: [usize; 2], lr: f64, gpu: bool) -> Result<f32>
    fn generate(&mut self, prompt: &[u8], max_len: usize, temperature: f32, gpu: bool) -> Result<Vec<u8>>

    fn model_config(&self) -> Option<&ModelConfig>
    fn training_config(&self) -> Option<&TrainingConfig>
    fn patching_config(&self) -> Option<&PatchingConfig>
}
```

### JuliaEngineBuilder

Fluent API for engine initialization.

```rust
impl JuliaEngineBuilder {
    fn new() -> Self
    fn config(self, path: impl Into<String>) -> Self
    fn model(self, name: impl Into<String>) -> Self
    fn module(self, path: impl Into<String>) -> Self
    fn with_patching(self, enabled: bool) -> Self
    fn build(self) -> Result<JuliaEngine>
}
```

### Config Types

```rust
struct ModelConfig {
    vocab: u32,
    hidden: u32,
    ffn: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    experts: u32,
    topk: u32,
    loops: u32,
    ctx_len: u32,
    rope_base: f64,
    r: u32,
}

struct TrainingConfig {
    batch_size: u32,
    learning_rate: f64,
    warmup_steps: u32,
    max_steps: u32,
    grad_clip: f64,
}

struct PatchingConfig {
    enabled: bool,
    patcher_hidden: u32,
    threshold: f32,
    ema_momentum: f32,
}

struct FullConfig {
    models: HashMap<String, ModelConfig>,
    training: TrainingConfig,
    patching: PatchingConfig,
}
```

## Examples

Run examples:

```bash
# Inference
cargo run --example inference

# Training with dynamic patching
cargo run --example training
```

## Performance

| Operation | CPU | GPU (A100) |
|-----------|-----|------------|
| Forward (1.9B) | ~100ms | ~10ms |
| Training step | ~500ms | ~50ms |
| Generation (100 tokens) | ~10s | ~1s |

With dynamic patching: **50% FLOP reduction** during inference.

## Architecture

```
Rust FFI (ffi_julia)
    ↓ jlrs
Julia Runtime
    ↓
MLCore.jl (BLT MoE Transformer)
    ↓ CUDA.jl
GPU Kernels
```

## Error Handling

All functions return `Result<T>` with descriptive error messages. Use `?` operator for propagation or pattern matching for custom handling.

## Dependencies

- `jlrs` - Julia runtime integration
- `serde` / `serde_yaml` - Config deserialization
- `anyhow` - Error handling

## References

- MLCore.jl: [../julia/](../julia/)
- Config format: [../julia/config.yaml](../julia/config.yaml)
- jlrs: [https://github.com/Taaitaaiger/jlrs](https://github.com/Taaitaaiger/jlrs)
