# 学習システム設計

## 概要

MoE Transformer (6.9B total / 1.8B active) の学習パイプライン設計。
Forward → Loss → Backward → Optimizer の完全実装。
**Rust + Go + Python** のマルチ言語実装。

---

## 実装状況

| コンポーネント | Rust | Go | Python | CUDA (共有) |
|----------------|------|-----|--------|-------------|
| Tensor | ✅ 型定義 + Error型 | ✅ Shape + Tensor | ✅ numpy backend | - |
| Embedding | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ scatter_add |
| RMSNorm | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| Linear (GEMM) | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| MQA Attention | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| RoPE (NTK) | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| MoE Router | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ softmax_topk |
| Expert FFN | ✅ SwiGLU | ✅ SwiGLU | ✅ SwiGLU | ✅ SiLU + GEMM |
| CrossEntropyLoss | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| AuxLoss | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| AdamW | ✅ 実装 | ✅ 実装 | ✅ 実装 | ✅ カーネル |
| **デコード** | | | | |
| Argmax | ✅ API | ✅ API | ✅ API | ✅ カーネル |
| Sample | ✅ API | ✅ API | ✅ API | ✅ カーネル |
| TopK Sample | ✅ API | ✅ API | ✅ API | ✅ カーネル |
| TopP Sample | ✅ API | ✅ API | ✅ API | ✅ カーネル |
| **最適化** | | | | |
| Gradient Checkpoint | ✅ | - | - | - |
| Mixed Precision | ✅ | - | - | - |
| CUDA Graph | ✅ | - | - | - |
| **GPU Trainer** | ✅ GpuTrainer | - | - | - |

---

## プロジェクト構成

```
machine_learning/
├── crates/
│   ├── cuda/           # 共有CUDAカーネル
│   │   ├── kernels/    # .cu ファイル (9個)
│   │   └── src/        # Rust FFI + stub.c
│   ├── rust/           # Rust実装
│   │   ├── nn-core/    # モデル・学習ロジック
│   │   └── nn-ffi/     # CUDA FFI ブリッジ
│   ├── go/             # Go実装
│   │   ├── tensor/     # Tensor操作
│   │   ├── cuda/       # cgo CUDAバインディング
│   │   ├── layer/      # NN層
│   │   ├── model/      # MoEモデル
│   │   └── train/      # 学習パイプライン
│   └── python/         # Python実装
│       ├── nn/         # NNモジュール (tensor, layers, model, train)
│       ├── cuda/       # ctypes CUDAバインディング
│       └── tests/      # pytest テスト
└── docs/
```

---

## 学習パイプライン全体像

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
║              Activations 保存 (for backward)           ║
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
║  各層で grad 計算、パラメータに蓄積                    ║
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

## CUDA カーネル一覧 (共有)

### 実装済みカーネル

| ファイル | カーネル | 機能 |
|----------|----------|------|
| elementwise.cu | `cuda_silu` | SiLU activation (x * sigmoid(x)) |
| | `cuda_add` | Element-wise addition |
| | `cuda_mul` | Element-wise multiplication |
| | `cuda_scale` | Scalar multiplication |
| softmax.cu | `cuda_softmax` | Row-wise softmax |
| | `cuda_softmax_topk` | Softmax + top-k (router用) |
| rmsnorm.cu | `cuda_rmsnorm` | RMSNorm |
| | `cuda_rmsnorm_residual` | Fused RMSNorm + residual |
| gemm.cu | `cuda_gemm` | GEMM (32x32 tiling) |
| | `cuda_gemm_beta` | GEMM with accumulation |
| | `cuda_batched_gemm` | Batched GEMM |
| rope.cu | `cuda_rope_freqs` | NTK RoPE 周波数計算 |
| | `cuda_rope_forward` | RoPE 適用 |
| | `cuda_rope_qk` | Q/K 同時 RoPE |
| attention.cu | `cuda_attention_scores` | Q @ K^T * scale |
| | `cuda_attention_output` | weights @ V |
| | `cuda_flash_attention` | FlashAttention風 fused |
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

### アーキテクチャサポート

```
sm_70: Volta (V100)
sm_75: Turing (RTX 20xx)
sm_80: Ampere (A100)
sm_86: Ampere (RTX 30xx)
sm_89: Ada (RTX 40xx)
sm_90: Hopper (H100)
```

### 言語別ビルド

```bash
# Rust (Cargo経由で自動ビルド)
cargo build --release

# Go (Makefile経由)
cd crates/go/cuda && make

# Python (pip経由)
cd crates/python && pip install -e ".[dev]"
```

---

## 1. Loss 設計

### 1.1 Cross Entropy Loss

```
L_ce = -1/N * Σ log(softmax(logits)[target])

実装:
  1. logits: (B, T, V) - batch, seq, vocab
  2. log_softmax: numerically stable版
  3. NLLLoss: gather + mean reduction
```

### 1.2 MoE Auxiliary Loss (Load Balancing)

```
目的: Expert の利用率を均等化

L_aux = α * Σ_i (f_i * P_i)

where:
  f_i = (tokens routed to expert i) / total_tokens
  P_i = mean(router_probs for expert i)
  α = aux_loss_weight (typically 0.01)

理想: 全 expert が均等に使われる → f_i ≈ 1/n_experts
```

### 1.3 Total Loss

```
L_total = L_ce + α * L_aux

α = 0.01 (default, tunable)
```

---

## 2. Backward Pass 設計

### 2.1 Autograd 方針

**採用: 手動実装 (教育目的 + 完全制御)**

各Layerに `forward()` と `backward()` を実装。

### 2.2 各層の Backward

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

## 3. Optimizer 設計

### 3.1 AdamW

```
標準的な AdamW:

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

学習時のトークン生成をGPU上で完結させ、CPU↔GPU転送を最小化。

#### Rust実装

```rust
// crates/rust/nn-ffi/src/trainer.rs

pub enum DecodingStrategy {
    Greedy,                           // argmax
    Sample { temperature: f32 },      // multinomial
    TopK { k: i32, temperature: f32 },
    TopP { top_p: f32, temperature: f32 },
}
```

#### Go実装

```go
// crates/go/cuda/cuda.go

func Argmax(logits []float32, output []int32, ...) error
func Sample(logits []float32, output []int32, seeds []uint64, ...) error
func TopKSample(logits []float32, output []int32, ...) error
func TopPSample(logits []float32, output []int32, ...) error
```

#### データフロー

```
┌─────────────────────────────────────────────────────────┐
│  Training Loop (GPU上で完結)                            │
│                                                         │
│  input_tokens → Forward → logits                        │
│       ↓                      ↓                          │
│  (GPU resident)     decode() → next_tokens (GPU)        │
│                              ↓                          │
│                     次ステップの入力として再利用          │
│                              ↓                          │
│  get_loss() ────────────→ loss (CPU転送: 唯一)          │
└─────────────────────────────────────────────────────────┘
```

#### CUDAカーネル詳細

| カーネル | 機能 | アルゴリズム |
|----------|------|-------------|
| `cuda_argmax` | Greedy decoding | Warp reduction |
| `cuda_sample` | Multinomial sampling | LCG RNG + CDF search |
| `cuda_topk_sample` | Top-k sampling | Partial sort + sample |
| `cuda_topp_sample` | Nucleus sampling | Sorted probs + cumsum threshold |

---

## 4. メモリ最適化

### 4.1 Gradient Checkpointing

```
問題: Activations保存でメモリ爆発
解決: 一部のActivationsを再計算

戦略:
  - 各 Transformer Block の入力のみ保存
  - Backward時に Block 内を再計算
  - メモリ: O(layers) vs O(layers × seq_len × hidden)
```

### 4.2 Mixed Precision (FP16/BF16)

```
Forward/Backward: FP16/BF16 で計算
Master weights: FP32 で保持
Gradient accumulation: FP32

Loss scaling: Dynamic loss scaling で underflow 防止
```

### 4.3 Gradient Accumulation

```
小バッチ × 複数回 → 大きな effective batch size

for micro_batch in micro_batches:
    loss = forward(micro_batch)
    loss.backward()  # grad 蓄積
optimizer.step()     # accumulation後に更新
optimizer.zero_grad()
```

---

## 5. 学習ループ実装

### Rust実装

```rust
// crates/rust/nn-core/src/train.rs

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

### Go実装

```go
// crates/go/train/trainer.go

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

### Python実装

```python
# crates/python/nn/train.py

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

## 6. 実装完了状況

### Phase 1: CUDA カーネル追加 ✅ 完了
- [x] CrossEntropyLoss (forward + backward)
- [x] AuxLoss (MoE load balancing)
- [x] AdamW optimizer kernel
- [x] Gradient clipping kernel
- [x] ScatterAdd (Embedding backward)

### Phase 2: Rust ↔ CUDA 連携 ✅ 完了
- [x] nn-ffi crate 作成
- [x] DeviceBuffer (GPU メモリ管理)
- [x] GpuTensor (GPU テンソル)
- [x] 高レベル API (rmsnorm, gemm, silu, softmax, cross_entropy, adamw)

### Phase 3: 最適化 ✅ 完了
- [x] Gradient Checkpointing (nn-core/checkpoint.rs)
- [x] Mixed Precision (FP16/BF16) (nn-core/mixed_precision.rs)
- [x] CUDA Graph 最適化 (nn-ffi/cuda_graph.rs)

### Phase 4: GPU常駐学習 ✅ 完了
- [x] GPU Decode カーネル (argmax, sample, topk, topp)
- [x] GpuTrainer (nn-ffi/trainer.rs)
- [x] DecodingStrategy (Greedy/Sample/TopK/TopP)
- [x] 最小限CPU転送設計

### Phase 5: Go実装 ✅ 完了
- [x] tensor パッケージ (Shape, DType, Tensor)
- [x] cuda パッケージ (cgo バインディング + Makefile)
- [x] layer パッケージ (Embedding, RMSNorm, Linear, SwiGLU)
- [x] model パッケージ (Attention, Router, MoE, Transformer)
- [x] train パッケージ (Trainer, AdamW, LRスケジューラ)

### Phase 6: Python実装 ✅ 完了
- [x] nn.tensor モジュール (numpy backend, DType)
- [x] nn.layers モジュール (Embedding, RMSNorm, Linear, SwiGLU)
- [x] nn.model モジュール (Attention, Router, MoE, Transformer)
- [x] nn.train モジュール (Trainer, AdamW, LRスケジューラ)
- [x] cuda パッケージ (ctypes バインディング + CPU fallback)

---

## 決定事項

- [x] 学習まで実装
- [x] Loss: CrossEntropy + MoE Aux Loss
- [x] Optimizer: AdamW
- [x] 手動 backward 実装（教育目的）
- [x] **CUDA: 全カーネル実装完了** (Phase 1)
- [x] **Rust: nn-ffi連携完了** (Phase 2)
- [x] **Rust: Phase 3最適化完了**
- [x] **Rust: Phase 4 GPU常駐学習完了**
- [x] **Go: Phase 5実装完了**
- [x] **Python: Phase 6実装完了**
- [ ] 分散学習: 対象外

---

## テスト状況

| 言語 | パッケージ | テスト数 | 状態 |
|------|-----------|----------|------|
| Rust | nn-core | 34 | ✅ |
| Rust | nn-cuda | 2 | ✅ |
| Rust | nn-ffi | 17 | ✅ |
| **Rust計** | | **53** | ✅ |
| Go | tensor | 15 | ✅ |
| Go | model | 11 | ✅ |
| Go | train | 5 | ✅ |
| **Go計** | | **31** | ✅ |
| Python | tensor | 18 | ✅ |
| Python | model | 16 | ✅ |
| Python | train | 8 | ✅ |
| **Python計** | | **42** | ✅ |
| **総計** | | **126** | ✅ |

---

## 議論メモ

- 学習パイプライン全体を設計
- 各層の backward を明示的に定義
- MoE 固有の Aux Loss (Load Balancing) を含む
- **CUDA カーネル共有**:
  - Rust: FFI経由 (build.rs)
  - Go: cgo経由 (Makefile)
  - Python: ctypes経由 (CPU fallback付き)
- **nn-cuda 全カーネル実装完了**:
  - Forward: elementwise, softmax, rmsnorm, gemm, rope, attention
  - Training: loss (CE + AuxLoss), optimizer (AdamW, grad_clip, scatter_add)
  - Decode: argmax, sample, topk_sample, topp_sample
- CUDA 未対応環境では stub.c でリンク可能
- **Go実装完了**:
  - tensor: Shape, DType, Tensor (matmul, softmax, silu等)
  - layer: Embedding, RMSNorm, Linear, SwiGLU
  - model: MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
  - train: Trainer, AdamW, LR schedule
  - cuda: cgo bindings (Makefile for standalone build)
- **Python実装完了**:
  - nn.tensor: numpy backend, DType enum, Tensor ops
  - nn.layers: Embedding, RMSNorm, Linear, SwiGLU
  - nn.model: Config, MQAttention, Router, MoELayer, TransformerBlock, MoETransformer
  - nn.train: TrainConfig, Trainer, AdamW, LR schedule
  - cuda: ctypes bindings with CPU fallback for all operations
