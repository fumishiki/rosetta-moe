# MLCore.jl

BLT Recurrent MoE Transformer (1.9B parameters, 232M active) implementation.

## Architecture Overview

| Component | Details |
|-----------|---------|
| **Model** | BLT + Recurrent (4-loop) + BLT |
| **Total Params** | ~1.9B |
| **Active Params** | ~232M |
| **Tokenizer** | Byte-Level (vocab_size=256, no training required) |
| **Position Encoding** | RoPE + Timestep Encoding |
| **Normalization** | Derf (Dynamic erf, parameterless) |
| **MoE Structure** | Shared Expert + Low-Rank Delta (r=256) |
| **Experts** | 128 (top-k=8) |
| **Optimizer** | Muon (hidden layers) + AdamW (Embedding/Router/LM Head) |
| **Precision** | INT4/NF4 (Julia custom implementation, hardware-agnostic) |

## Structure

```
julia/
├── Project.toml             # Dependencies
├── src/ (893 lines, 41% reduction)
│   ├── MLCore.jl           # Module entry (34 lines)
│   ├── layers.jl           # Model architecture (306 lines)
│   ├── training.jl         # Data + Optimizer + Training (227 lines)
│   └── quantization.jl     # INT4/NF4 + Mixed-Precision (326 lines)
├── test/ (114 lines, 72% reduction)
│   ├── runtests.jl         # Test runner (5 lines)
│   └── test_all.jl         # All tests (109 lines)
├── example.jl              # Example usage (58 lines, 59% reduction)
└── README.md
```

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| Lux.jl | Neural network layers | 1.x |
| CUDA.jl | GPU kernels | 5.x |
| Zygote.jl | Automatic differentiation | 0.6.x |
| Optimisers.jl | Optimizer utilities | 0.3.x |

## Configuration

All model settings are in [config.yaml](./config.yaml). Edit this file to customize:

```yaml
models:
  1.9B:
    hidden: 2048      # Hidden dimension
    experts: 128      # Number of experts
    loops: 4          # Recurrent iterations
    # ... more settings

training:
  batch_size: 256000  # Total tokens per batch
  learning_rate: 1e-3
  # ...

patching:
  enabled: false      # Enable dynamic patching
  threshold: 2.5
```

## Quick Start

### Installation

```bash
cd lang/julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### Basic Usage

```julia
using Pkg
Pkg.activate(".")

using MLCore

# Load configuration from YAML (recommended)
config = load_config("1.9B")

# Or use built-in defaults
config = C1_9B

# Create model
model = BLTRecurrentMoE(config)

# With dynamic patching
model = BLTRecurrentMoEWithPatching(config, use_patching=true, threshold=2.5f0)

# Forward pass
input_bytes = [72, 101, 108, 108, 111]  # "Hello" (0-indexed)
logits = model(reshape(input_bytes, 1, length(input_bytes)))

# Create optimizer
optimizer = HybridOptimizer(
    muon_lr=1e-3,
    adamw_lr=1e-3
)

# Prepare data
text = "Hello, world!"
bytes = text_to_bytes(text)
batches = create_byte_batches(UInt8.(bytes), 10, 1)

# Train
train_model!(model, batches, optimizer; epochs=1)

# Generate
prompt = [72, 101, 108, 108, 111]  # "Hello"
generated = generate_text(model, prompt, 100)
println(bytes_to_text(generated))
```

### Custom Model Sizes

Add new configurations to `config.yaml`:

```yaml
models:
  custom:
    vocab: 256
    hidden: 3072
    ffn: 24576
    heads: 48
    kv_heads: 4
    head_dim: 64
    experts: 128
    topk: 8
    loops: 4
    ctx_len: 8192
    rope_base: 10000.0
    r: 384
```

Then load:

```julia
config = load_config("custom")
model = BLTRecurrentMoE(config)
```

### Training with YAML Config

```julia
# Load full configuration
full_cfg = load_full_config()

# Extract settings
model_cfg = load_config("1.9B")
training_cfg = full_cfg["training"]
patching_cfg = full_cfg["patching"]

# Create model with patching settings
model = BLTRecurrentMoEWithPatching(
    model_cfg,
    use_patching=patching_cfg["enabled"],
    threshold=Float32(patching_cfg["threshold"]),
    ema_momentum=Float32(patching_cfg["ema_momentum"])
)

# Train with settings from YAML
train!(model, data,
       epochs=10,
       lr=training_cfg["learning_rate"],
       gpu=true)
```

### Run Example

```bash
julia --project example.jl
```

### Quantization Usage

```julia
using Pkg
Pkg.activate(".")

include("src/MLCore.jl")
using .MLCore

# Create standard linear layer
weight = randn(Float32, 512, 1024)
layer = LinearLayer(1024, 512, Float32; bias=false)
layer.weight .= weight

# Quantize to INT4 (with Hadamard transform)
layer_int4 = QuantizedLinearINT4(weight; use_hadamard=true, hadamard_group_size=32)

# Quantize to NF4 (blockwise)
layer_nf4 = QuantizedLinearNF4(weight; blocksize=64)

# Forward pass
x = randn(Float32, 8, 1024)
output_int4 = layer_int4(x)
output_nf4 = layer_nf4(x)

# Estimate memory reduction
total_params = 512 * 1024
reduction = estimate_memory_reduction(total_params, total_params)
println("Memory reduction: $(round(reduction, digits=1))%")
```

### Test Quantization

```bash
julia --project test_quantization.jl
```

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| **Model Architecture** |
| BLT Input Layer | ✅ Complete | [layers.jl:367-382](src/layers.jl#L367-L382) |
| BLT Output Layer | ✅ Complete | [layers.jl:384-399](src/layers.jl#L384-L399) |
| Recurrent Block (4-loop) | ✅ Complete | [layers.jl:405-443](src/layers.jl#L405-L443) |
| MQA Attention + RoPE | ✅ Complete | [layers.jl:140-205](src/layers.jl#L140-L205) |
| Timestep Encoding | ✅ Complete | [layers.jl:408-414](src/layers.jl#L408-L414) |
| **Normalization** |
| Derf (Dynamic erf) | ✅ Complete | [layers.jl:42](src/layers.jl#L42) |
| **MoE** |
| Router | ✅ Complete | [layers.jl:212-252](src/layers.jl#L212-L252) |
| Shared Expert (SwiGLU) | ✅ Complete | [layers.jl:254-275](src/layers.jl#L254-L275) |
| Low-Rank Expert Delta | ✅ Complete | [layers.jl:277-315](src/layers.jl#L277-L315) |
| MoE Layer | ✅ Complete | [layers.jl:318-361](src/layers.jl#L318-L361) |
| **Optimizer** |
| Muon (hidden layers) | ✅ Complete | [optimizer.jl:6-46](src/optimizer.jl#L6-L46) |
| AdamW (other layers) | ✅ Complete | [optimizer.jl:51-94](src/optimizer.jl#L51-L94) |
| Hybrid Optimizer | ✅ Complete | [optimizer.jl:99-185](src/optimizer.jl#L99-L185) |
| **Training** |
| Cross-entropy loss | ✅ Complete | [training.jl:5-27](src/training.jl#L5-L27) |
| Training loop | ✅ Complete | [training.jl:95-137](src/training.jl#L95-L137) |
| Text generation | ✅ Complete | [training.jl:143-196](src/training.jl#L143-L196) |
| **Quantization** |
| Hadamard Transform | ✅ Complete | [quantization.jl:22-61](src/quantization.jl#L22-L61) |
| INT4 Quantization | ✅ Complete | [quantization.jl:66-82](src/quantization.jl#L66-L82) |
| NF4 Quantization | ✅ Complete | [quantization.jl:87-129](src/quantization.jl#L87-L129) |
| Quantized Linear (INT4) | ✅ Complete | [quantization.jl:134-189](src/quantization.jl#L134-L189) |
| Quantized Linear (NF4) | ✅ Complete | [quantization.jl:194-228](src/quantization.jl#L194-L228) |

## Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Architecture | BLT + Recurrent + BLT | Ultra-lightweight (1.9B total, 232M active) |
| Tokenizer | Byte-Level (vocab_size=256) | No training required, universal |
| Position Encoding | RoPE + Timestep | Standard + loop identification |
| Normalization | Derf | Parameterless, faster than RMSNorm |
| MoE | Shared Expert + Low-Rank Delta | Efficient sparse learning |
| Optimizer | Muon + AdamW hybrid | Muon for hidden layers, AdamW for others |
| Recurrent Loops | 4 loops | Stability vs performance balance |
| Batch Size | 256K tokens | Optimal for 1.9B model |
| Quantization | INT4/NF4 | 70% memory reduction, hardware-agnostic |

## References

- Model specification: [../../docs/1-model.md](../../docs/1-model.md)
- Architecture: [../../docs/architecture.md](../../docs/architecture.md)
- CLAUDE.md: `~/.claude/CLAUDE.md`
