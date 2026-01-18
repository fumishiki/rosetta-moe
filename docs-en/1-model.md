# MoE Transformer Design Document

## Overview

Design specifications for the 6.9B MoE Transformer (Mixture of Experts).
Multi-language implementation in **Rust + Go + Python + CUDA**.

---

## Decisions

- [x] Architecture: **MoE Transformer (6.9B total, ~1.8B active)**
- [x] Training: **Supported (forward + backward + optimizer)**
- [x] Tokenizer: **SentencePiece (self-trained, Apache 2.0)**
- [x] Weight Tying: **No (Embedding / LM Head separated)**
- [x] Position Encoding: **NTK RoPE (32K train → 256K inference)**
- [x] Implementation: **Rust + Go + Python all completed**
- [x] GPU Decode: **argmax, sample, top-k, top-p implemented**
- [x] Type-level design: **TensorError, TensorResult introduced**

---

## MoE Transformer Specifications

### Benefits of MoE (Mixture of Experts)

```
Dense Transformer:
  All parameters computed every time → 6.9B params = 6.9B active

MoE Transformer:
  Experts selectively activated → 6.9B params, 1.8B active per token
  Computational efficiency: ~3.8x (theoretical)
```

### Model Parameters

| Parameter | Mixtral 8x7B | DeepSeek-MoE | Ours |
|-----------|--------------|--------------|------|
| total_params | 46.7B | 16B | **6.9B** |
| active_params | 12.9B | 2.8B | **~1.8B** |
| hidden_dim | 4096 | 2048 | **768** |
| n_layers | 32 | 28 | **30** |
| n_heads | 32 | 16 | **12** |
| n_kv_heads | 8 (GQA) | 16 | **1 (MQA)** |
| n_experts | 8 | 64 | **16** |
| top_k_experts | 2 | 6 | **4** |
| vocab_size | 32000 | 102400 | 32000 |
| context_len | 32768 | 4096 | **32K (→256K with NTK)** |
| FFN dim/expert | 14336 | 1408 | **6144** |
| head_dim | 128 | 128 | **64** |
| Norm | RMSNorm | RMSNorm | RMSNorm |
| Activation | SiLU | SiLU | SiLU |
| Position | RoPE | RoPE | **NTK RoPE** |

### Parameter Calculation

```
Embedding:        32000 × 768            =   24.6M
Per Layer:
  - Attention:    768×768×2 + 768×64×2   =    1.3M (Q,O + K,V MQA)
  - Router:       768 × 16               =   12K
  - Expert FFN:   768 × 6144 × 3 × 16    =  226.5M (gate,up,down × 16 experts)
  - Norms:        768 × 2                =    1.5K
  Layer Total:                           ≈  227.8M

Total: 24.6M + (227.8M × 30) + 24.6M (LM head) ≈ 6.9B
Active per token: 24.6M + (1.3M + 56.6M) × 30 + 24.6M ≈ 1.8B
```

---

## Architecture

```
Input Token IDs
    ↓
Embedding (32000 × 768)
    ↓
╔══════════════════════════════════════╗
║     MoE Transformer Block × 30       ║
║                                      ║
║  RMSNorm                             ║
║      ↓                               ║
║  MQA Attention + RoPE                ║
║    - Q: 768 → 768 (12 heads)         ║
║    - K,V: 768 → 64 (1 KV head)       ║
║      ↓                               ║
║  + Residual                          ║
║      ↓                               ║
║  RMSNorm                             ║
║      ↓                               ║
║  MoE Layer (16 Experts, top-k=4)     ║
║    Router → [E0..E15] → Mix          ║
║      ↓                               ║
║  + Residual                          ║
╚══════════════════════════════════════╝
    ↓
RMSNorm
    ↓
LM Head (768 × 32000)
    ↓
Output Logits
```

### Expert FFN (SwiGLU)

```
x → W_gate → SiLU ─┐
                   ⊙ → W_down → out
x → W_up ──────────┘

Dims: 768 → 6144 → 768
```

---

## CUDA Kernel List

| Kernel | Priority | Difficulty | Status | Notes |
|--------|----------|------------|--------|-------|
| GEMM (MatMul) | Required | High | ✅ | 32x32 tiling |
| RMSNorm | Required | Low | ✅ | reduction kernel |
| SiLU | Required | Low | ✅ | element-wise |
| RoPE | Required | Medium | ✅ | NTK scaling support |
| Softmax | Required | Medium | ✅ | numerically stable |
| GQA Attention | Required | High | ✅ | FlashAttention-style fused |
| Embedding | Required | Low | ✅ | gather kernel |
| MoE Router | Required | Medium | ✅ | softmax + top-k |
| CrossEntropy | Training | Medium | ✅ | forward + backward |
| Aux Loss | Training | Medium | ✅ | load balancing |
| AdamW | Training | Medium | ✅ | fused optimizer |
| Grad Clip | Training | Medium | ✅ | global norm |
| **Decode** | | | | |
| Argmax | Inference | Low | ✅ | greedy decoding |
| Sample | Inference | Medium | ✅ | multinomial + temp |
| TopK Sample | Inference | Medium | ✅ | top-k sampling |
| TopP Sample | Inference | Medium | ✅ | nucleus sampling |

---

## Tokenizer / Embedding

### Tokenizer

| Item | Value |
|------|-------|
| Method | SentencePiece (self-trained) |
| Algorithm | Unigram or BPE |
| vocab_size | 32000 |
| Special tokens | `<pad>`, `<unk>`, `<bos>`, `<eos>` |
| License | Apache 2.0 |

**Training data candidates:**
- Wikipedia (Japanese + English)
- CC-100 (CommonCrawl)

**Training code example:**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='unigram',
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    character_coverage=0.9995,
)
```

### Embedding Layer

| Item | Value |
|------|-------|
| vocab_size | 32000 |
| hidden_dim | 2048 |
| Parameters | 65.5M |
| Weight Tying | No |
| Initialization | Normal(0, 0.02) |

### LM Head

| Item | Value |
|------|-------|
| input_dim | 2048 |
| output_dim | 32000 |
| Parameters | 65.5M |
| bias | No |

---

## MoE Technical Points

1. **Router** — Softmax + Top-K selection
2. **Expert Dispatch** — Route tokens to appropriate experts
3. **Expert Combine** — Aggregate weighted outputs
4. **Load Balancing Loss** — Equalize expert utilization (during training)
5. **Capacity Factor** — Drop strategy for overloaded experts

---

## NTK RoPE (Position Encoding)

### Overview

```
Traditional RoPE:
  Performance degrades beyond training context_len

NTK-aware RoPE:
  Scale base frequency for long context support
  Extend context_len by α times without training
```

### Design

| Item | Value |
|------|-------|
| Training context_len | 32K |
| NTK scale α | 8 |
| Inference context_len | **256K** (32K × 8) |
| base frequency | 10000 → 10000 × α^(d/(d-2)) |

### Implementation

```python
# NTK RoPE scaling
def ntk_rope_freqs(dim: int, base: float = 10000, alpha: float = 8.0):
    # NTK-aware interpolation
    base = base * alpha ** (dim / (dim - 2))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    return freqs
```

### Benefits

1. **Training cost reduction** — Train at 32K, infer at 256K
2. **No additional training** — Extension through scaling only
3. **Quality preservation** — Less performance degradation at long context

---

## Optimization Levels

| Level | Content |
|-------|---------|
| L1 | Naive CUDA implementation |
| L2 | Shared memory tiling |
| L3 | FlashAttention, Tensor Core |
| L4 | Quantization (INT8/INT4) |
