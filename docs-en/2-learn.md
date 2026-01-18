# Training System Design

## Overview

Training pipeline design for MoE Transformer (6.9B total / 1.8B active).
Complete implementation of Forward → Loss → Backward → Optimizer.
Multi-language implementation in **Rust + Go + Python**.

---

## Implementation Status

| Component | Rust | Go | Python | CUDA (Shared) |
|-----------|------|-----|--------|---------------|
| Tensor | ✅ Type def + Error type | ✅ Shape + Tensor | ✅ numpy backend | - |
| Embedding | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ scatter_add |
| RMSNorm | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| Linear (GEMM) | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| MQA Attention | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| RoPE (NTK) | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| MoE Router | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ softmax_topk |
| Expert FFN | ✅ SwiGLU | ✅ SwiGLU | ✅ SwiGLU | ✅ SiLU + GEMM |
| CrossEntropyLoss | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| AuxLoss | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| AdamW | ✅ Implemented | ✅ Implemented | ✅ Implemented | ✅ Kernel |
| **Decode** | | | | |
| Argmax | ✅ API | ✅ API | ✅ API | ✅ Kernel |
| Sample | ✅ API | ✅ API | ✅ API | ✅ Kernel |
| TopK Sample | ✅ API | ✅ API | ✅ API | ✅ Kernel |
| TopP Sample | ✅ API | ✅ API | ✅ API | ✅ Kernel |
| **Optimization** | | | | |
| Gradient Checkpoint | ✅ | - | - | - |
| Mixed Precision | ✅ | - | - | - |
| CUDA Graph | ✅ | - | - | - |
| **GPU Trainer** | ✅ GpuTrainer | - | - | - |

---

## Project Structure

```
machine_learning/
├── rust/               # Rust implementation
│   ├── nn-core/        # Model, training logic
│   └── nn-ffi/         # CUDA FFI bridge
├── go/                 # Go implementation
│   ├── tensor/         # Tensor operations
│   ├── cuda/           # cgo CUDA bindings
│   ├── layer/          # NN layers
│   ├── model/          # MoE model
│   └── train/          # Training pipeline
├── python/             # Python implementation
│   ├── nn/             # NN modules
│   ├── cuda/           # ctypes CUDA bindings
│   └── tests/          # pytest tests
├── cuda/               # Shared CUDA kernels
│   ├── kernels/        # .cu files (9)
│   └── src/            # stub.c (CPU fallback)
├── docs-jp/            # Japanese documentation
└── docs-en/            # English documentation
```

---

## Training Pipeline Overview

```
[Data Loader]
    ↓
[Tokenizer] → Token IDs (batch_size × seq_len)
    ↓
╔═══════════════════════════════════════════════════════╗
║                    Forward Pass                        ║
║                                                        ║
║  Input → Embedding → Blocks×30 → LM Head → Logits    ║
║                         ↓                              ║
║              Save activations (for backward)           ║
╚═══════════════════════════════════════════════════════╝
    ↓
╔═══════════════════════════════════════════════════════╗
║                    Loss Computation                    ║
║                                                        ║
║  Logits + Labels → CrossEntropyLoss                   ║
║                  + MoE Aux Loss (Load Balance)        ║
║                  → Total Loss                          ║
╚═══════════════════════════════════════════════════════╝
    ↓
╔═══════════════════════════════════════════════════════╗
║                    Backward Pass                       ║
║                                                        ║
║  Loss → dLogits → dBlocks → dEmbedding → Gradients   ║
║                                                        ║
║  Compute grad at each layer, accumulate to params     ║
╚═══════════════════════════════════════════════════════╝
    ↓
╔═══════════════════════════════════════════════════════╗
║                    Optimizer Step                      ║
║                                                        ║
║  AdamW: param -= lr * (m / (sqrt(v) + eps) + wd * p)  ║
║                                                        ║
║  Gradient Clipping → Update → Zero Grad               ║
╚═══════════════════════════════════════════════════════╝
    ↓
[Next Iteration]
```

---

## CUDA Kernel List (Shared)

### Implemented Kernels

| File | Kernel | Function |
|------|--------|----------|
| elementwise.cu | `cuda_silu` | SiLU activation (x * sigmoid(x)) |
| | `cuda_add` | Element-wise addition |
| | `cuda_mul` | Element-wise multiplication |
| | `cuda_scale` | Scalar multiplication |
| softmax.cu | `cuda_softmax` | Row-wise softmax |
| | `cuda_softmax_topk` | Softmax + top-k (for router) |
| rmsnorm.cu | `cuda_rmsnorm` | RMSNorm |
| | `cuda_rmsnorm_residual` | Fused RMSNorm + residual |
| gemm.cu | `cuda_gemm` | GEMM (32x32 tiling) |
| | `cuda_gemm_beta` | GEMM with accumulation |
| | `cuda_batched_gemm` | Batched GEMM |
| rope.cu | `cuda_rope_freqs` | NTK RoPE frequency calculation |
| | `cuda_rope_forward` | RoPE application |
| | `cuda_rope_qk` | Simultaneous Q/K RoPE |
| attention.cu | `cuda_attention_scores` | Q @ K^T * scale |
| | `cuda_attention_output` | weights @ V |
| | `cuda_flash_attention` | FlashAttention-style fused |
| loss.cu | `cuda_cross_entropy_forward` | CrossEntropy + log_probs |
| | `cuda_cross_entropy_backward` | softmax - one_hot |
| | `cuda_aux_loss_forward` | MoE load balancing |
| optimizer.cu | `cuda_adamw_step` | Fused AdamW update |
| | `cuda_zero_grad` | Zero gradients |
| | `cuda_grad_clip` | Global norm clipping |
| | `cuda_scatter_add` | Embedding backward |
| decode.cu | `cuda_argmax` | Greedy decoding |
| | `cuda_sample` | Multinomial sampling |
| | `cuda_topk_sample` | Top-k sampling |
| | `cuda_topp_sample` | Nucleus (top-p) sampling |

### Architecture Support

```
sm_70: Volta (V100)
sm_75: Turing (RTX 20xx)
sm_80: Ampere (A100)
sm_86: Ampere (RTX 30xx)
sm_89: Ada (RTX 40xx)
sm_90: Hopper (H100)
```

### Language-specific Build

```bash
# Rust (auto-build via Cargo)
cargo build --release

# Go (via Makefile)
cd go/cuda && make

# Python (via pip)
cd python && pip install -e ".[dev]"
```

---

## 1. Loss Design

### 1.1 Cross Entropy Loss

```
L_ce = -1/N * Σ log(softmax(logits)[target])

Implementation:
  1. logits: (B, T, V) - batch, seq, vocab
  2. log_softmax: numerically stable version
  3. NLLLoss: gather + mean reduction
```

### 1.2 MoE Auxiliary Loss (Load Balancing)

```
Purpose: Equalize expert utilization

L_aux = α * Σ_i (f_i * P_i)

where:
  f_i = (tokens routed to expert i) / total_tokens
  P_i = mean(router_probs for expert i)
  α = aux_loss_weight (typically 0.01)

Ideal: All experts used equally → f_i ≈ 1/n_experts
```

### 1.3 Total Loss

```
L_total = L_ce + α * L_aux

α = 0.01 (default, tunable)
```

---

## 2. Backward Pass Design

### 2.1 Autograd Policy

**Adopted: Manual implementation (educational purpose + full control)**

Implement `forward()` and `backward()` for each layer.

### 2.2 Backward for Each Layer

#### Embedding Backward

```
forward: x[i] = W[token_id[i]]
backward: dW[token_id[i]] += dx[i]  (scatter_add)
```

#### RMSNorm Backward

```
forward: y = x * rsqrt(mean(x²) + eps) * gamma
backward:
  d_gamma = Σ (dy * x_normalized)
  dx = dy * gamma * d_rms_norm(x)
```

#### MQA Attention Backward

```
forward:
  Q = x @ W_q     (768 → 768, 12 heads × 64 dim)
  K = x @ W_k     (768 → 64,  1 KV head)
  V = x @ W_v     (768 → 64,  1 KV head)
  Q, K = apply_rope(Q, K)
  attn = softmax(Q @ K.T / sqrt(64))  # K broadcast
  out = (attn @ V) @ W_o

backward:
  dV = attn.T @ d_out
  d_attn = d_out @ V.T
  d_attn = softmax_backward(d_attn, attn)
  dQ = d_attn @ K / sqrt(64)
  dK = d_attn.T @ Q / sqrt(64)
  dx = linear_backward(dQ, dK, dV)
```

#### MoE Layer Backward

```
forward:
  router_probs = softmax(x @ W_router)    # [B, T, 16]
  indices, weights = top_k(router_probs, k=4)
  expert_outputs = dispatch_to_experts(x, indices)
  out = combine(expert_outputs, weights)

backward:
  d_expert_outputs, d_weights = combine_backward(d_out)
  dx_experts = dispatch_backward(d_expert_outputs, indices)
  d_router_probs = top_k_backward(d_weights, indices)
  dx += d_router_probs @ W_router.T
  dW_router = x.T @ d_router_probs
```

#### SwiGLU FFN Backward

```
forward:
  gate = silu(x @ W_gate)    # 768 → 6144
  up = x @ W_up              # 768 → 6144
  out = (gate * up) @ W_down # 6144 → 768

backward:
  d_gate_up = d_out @ W_down.T
  d_gate = d_gate_up * up
  d_up = d_gate_up * gate
  d_silu = silu_backward(d_gate, x @ W_gate)

  dW_gate = x.T @ d_silu
  dW_up = x.T @ d_up
  dW_down = (gate * up).T @ d_out
  dx = d_silu @ W_gate.T + d_up @ W_up.T
```

---

## 3. Optimizer Design

### 3.1 AdamW

```
Standard AdamW:

m = β1 * m + (1 - β1) * g
v = β2 * v + (1 - β2) * g²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

Hyperparams:
  lr = 1e-4 (or schedule)
  β1 = 0.9
  β2 = 0.999
  eps = 1e-8
  weight_decay = 0.1
```

### 3.2 Learning Rate Schedule

```
Warmup + Cosine Decay:

if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

Default config:
  warmup_steps = 1000
  total_steps = 100000
  min_lr = base_lr * 0.1
```

### 3.3 GPU Decode (Token Generation)

Complete token generation on GPU during training, minimizing CPU↔GPU transfers.

#### Rust Implementation

```rust
// rust/nn-ffi/src/trainer.rs

pub enum DecodingStrategy {
    Greedy,                           // argmax
    Sample { temperature: f32 },      // multinomial
    TopK { k: i32, temperature: f32 },
    TopP { top_p: f32, temperature: f32 },
}
```

#### Go Implementation

```go
// go/cuda/cuda.go

func Argmax(logits []float32, output []int32, ...) error
func Sample(logits []float32, output []int32, seeds []uint64, ...) error
func TopKSample(logits []float32, output []int32, ...) error
func TopPSample(logits []float32, output []int32, ...) error
```

#### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (completed on GPU)                       │
│                                                         │
│  input_tokens → Forward → logits                        │
│       ↓                      ↓                          │
│  (GPU resident)     decode() → next_tokens (GPU)        │
│                              ↓                          │
│                     Reuse as input for next step        │
│                              ↓                          │
│  get_loss() ────────────→ loss (CPU transfer: only one) │
└─────────────────────────────────────────────────────────┘
```

#### CUDA Kernel Details

| Kernel | Function | Algorithm |
|--------|----------|-----------|
| `cuda_argmax` | Greedy decoding | Warp reduction |
| `cuda_sample` | Multinomial sampling | LCG RNG + CDF search |
| `cuda_topk_sample` | Top-k sampling | Partial sort + sample |
| `cuda_topp_sample` | Nucleus sampling | Sorted probs + cumsum threshold |

---

## 4. Memory Optimization

### 4.1 Gradient Checkpointing

```
Problem: Memory explosion from saving activations
Solution: Recompute some activations

Strategy:
  - Save only input of each Transformer Block
  - Recompute within block during backward
  - Memory: O(layers) vs O(layers × seq_len × hidden)
```

### 4.2 Mixed Precision (FP16/BF16)

```
Forward/Backward: Compute in FP16/BF16
Master weights: Keep in FP32
Gradient accumulation: FP32

Loss scaling: Dynamic loss scaling to prevent underflow
```

### 4.3 Gradient Accumulation

```
Small batch × multiple times → large effective batch size

for micro_batch in micro_batches:
    loss = forward(micro_batch)
    loss.backward()  # accumulate grad
optimizer.step()     # update after accumulation
optimizer.zero_grad()
```

---

## 5. Training Loop Implementation

### Rust Implementation

```rust
// rust/nn-core/src/train.rs

pub(crate) struct TrainConfig {
    pub(crate) batch_size: usize,
    pub(crate) seq_len: usize,
    pub(crate) lr: f32,
    pub(crate) warmup_steps: usize,
    pub(crate) total_steps: usize,
    pub(crate) grad_clip: f32,
    pub(crate) aux_loss_weight: f32,
}

impl Trainer {
    pub(crate) fn train_step(&mut self, input: &Tensor, targets: &Tensor) -> f32 {
        let logits = self.model.forward(input);
        let ce_loss = CrossEntropyLoss::forward(&logits, targets);
        let grad = CrossEntropyLoss::backward(&logits, targets);
        self.model.backward(&grad);
        self.optimizer.step(&mut params);
        self.current_step += 1;
        ce_loss
    }
}
```

### Go Implementation

```go
// go/train/trainer.go

type TrainConfig struct {
    LR          float32
    Beta1       float32
    Beta2       float32
    WarmupSteps int
    TotalSteps  int
    GradClip    float32
    AuxAlpha    float32
}

func (t *Trainer) TrainStep(input, targets *tensor.Tensor) float32 {
    logits := t.model.Forward(input)
    loss := crossEntropyLoss(logits, targets)
    auxLoss := t.model.TotalAuxLoss(t.config.AuxAlpha)
    gradOutput := tensor.Ones(logits.Shape(), tensor.F32)
    t.model.Backward(gradOutput)
    // AdamW update...
    t.step++
    return loss + auxLoss
}
```

### Python Implementation

```python
# python/nn/train.py

@dataclass
class TrainConfig:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    total_steps: int = 100000
    grad_clip: float = 1.0
    aux_loss_alpha: float = 0.01

class Trainer:
    def train_step(self, input_ids: Tensor, targets: Tensor) -> float:
        logits = self.model.forward(input_ids)
        loss, grad_logits = self._compute_loss(logits, targets)
        aux_loss = self.model.total_aux_loss(self.config.aux_loss_alpha)
        self.model.backward(grad_logits)
        # AdamW update...
        self.step += 1
        return loss + aux_loss
```

---

## 6. Implementation Completion Status

### Phase 1: CUDA Kernel Addition ✅ Complete
- [x] CrossEntropyLoss (forward + backward)
- [x] AuxLoss (MoE load balancing)
- [x] AdamW optimizer kernel
- [x] Gradient clipping kernel
- [x] ScatterAdd (Embedding backward)

### Phase 2: Rust ↔ CUDA Integration ✅ Complete
- [x] nn-ffi crate creation
- [x] DeviceBuffer (GPU memory management)
- [x] GpuTensor (GPU tensor)
- [x] High-level API (rmsnorm, gemm, silu, softmax, cross_entropy, adamw)

### Phase 3: Optimization ✅ Complete
- [x] Gradient Checkpointing (nn-core/checkpoint.rs)
- [x] Mixed Precision (FP16/BF16) (nn-core/mixed_precision.rs)
- [x] CUDA Graph optimization (nn-ffi/cuda_graph.rs)

### Phase 4: GPU-Resident Training ✅ Complete
- [x] GPU Decode kernels (argmax, sample, topk, topp)
- [x] GpuTrainer (nn-ffi/trainer.rs)
- [x] DecodingStrategy (Greedy/Sample/TopK/TopP)
- [x] Minimal CPU transfer design

### Phase 5: Go Implementation ✅ Complete
- [x] tensor package (Shape, DType, Tensor)
- [x] cuda package (cgo bindings + Makefile)
- [x] layer package (Embedding, RMSNorm, Linear, SwiGLU)
- [x] model package (Attention, Router, MoE, Transformer)
- [x] train package (Trainer, AdamW, LR scheduler)

### Phase 6: Python Implementation ✅ Complete
- [x] nn.tensor module (numpy backend, DType)
- [x] nn.layers module (Embedding, RMSNorm, Linear, SwiGLU)
- [x] nn.model module (Attention, Router, MoE, Transformer)
- [x] nn.train module (Trainer, AdamW, LR scheduler)
- [x] cuda package (ctypes bindings + CPU fallback)

---

## Decisions

- [x] Implement training
- [x] Loss: CrossEntropy + MoE Aux Loss
- [x] Optimizer: AdamW
- [x] Manual backward implementation (educational purpose)
- [x] **CUDA: All kernels implemented** (Phase 1)
- [x] **Rust: nn-ffi integration complete** (Phase 2)
- [x] **Rust: Phase 3 optimization complete**
- [x] **Rust: Phase 4 GPU-resident training complete**
- [x] **Go: Phase 5 implementation complete**
- [x] **Python: Phase 6 implementation complete**
- [ ] Distributed training: Out of scope

---

## Test Status

| Language | Package | Test Count | Status |
|----------|---------|------------|--------|
| Rust | nn-core | 34 | ✅ |
| Rust | nn-cuda | 2 | ✅ |
| Rust | nn-ffi | 17 | ✅ |
| **Rust Total** | | **53** | ✅ |
| Go | tensor | 15 | ✅ |
| Go | model | 11 | ✅ |
| Go | train | 5 | ✅ |
| **Go Total** | | **31** | ✅ |
| Python | tensor | 18 | ✅ |
| Python | model | 16 | ✅ |
| Python | train | 8 | ✅ |
| **Python Total** | | **42** | ✅ |
| **Grand Total** | | **126** | ✅ |

---

## Discussion Notes

- Designed entire training pipeline
- Explicitly defined backward for each layer
- Includes MoE-specific Aux Loss (Load Balancing)
- **CUDA kernel sharing**:
  - Rust: via FFI (build.rs)
  - Go: via cgo (Makefile)
  - Python: via ctypes (with CPU fallback)
- **nn-cuda all kernels implemented**:
  - Forward: elementwise, softmax, rmsnorm, gemm, rope, attention
  - Training: loss (CE + AuxLoss), optimizer (AdamW, grad_clip, scatter_add)
  - Decode: argmax, sample, topk_sample, topp_sample
- Linkable with stub.c for non-CUDA environments
- **Go implementation complete**:
  - tensor: Shape, DType, Tensor (matmul, softmax, silu, etc.)
  - layer: Embedding, RMSNorm, Linear, SwiGLU
  - model: MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
  - train: Trainer, AdamW, LR schedule
  - cuda: cgo bindings (Makefile for standalone build)
- **Python implementation complete**:
  - nn.tensor: numpy backend, DType enum, Tensor ops
  - nn.layers: Embedding, RMSNorm, Linear, SwiGLU
  - nn.model: Config, MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
  - nn.train: TrainConfig, Trainer, AdamW, LR schedule
  - cuda: ctypes bindings with CPU fallback for all operations
