# MoE Transformer 設計書

## 概要

6.9B MoE Transformer (Mixture of Experts) の設計仕様。
**Rust + Go + Python + CUDA** のマルチ言語実装。

---

## 決定事項

- [x] アーキテクチャ: **MoE Transformer (6.9B total, ~1.8B active)**
- [x] 学習: **対応（forward + backward + optimizer）**
- [x] Tokenizer: **SentencePiece (自前訓練, Apache 2.0)**
- [x] Weight Tying: **しない (Embedding / LM Head 分離)**
- [x] Position Encoding: **NTK RoPE (32K学習 → 256K推論)**
- [x] 実装: **Rust + Go + Python 全言語完了**
- [x] GPU Decode: **argmax, sample, top-k, top-p 実装完了**
- [x] 型レベル設計: **TensorError, TensorResult 導入**

---

## MoE Transformer 仕様

### MoE (Mixture of Experts) の利点

```
Dense Transformer:
  全パラメータが毎回計算 → 6.9B params = 6.9B active

MoE Transformer:
  Expert を選択的に活性化 → 6.9B params, 1.8B active per token
  計算効率: 約3.8倍（理論上）
```

### モデルパラメータ

| パラメータ | Mixtral 8x7B | DeepSeek-MoE | Ours |
|------------|--------------|--------------|------|
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

### パラメータ計算

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

## アーキテクチャ

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

## CUDAカーネル一覧

| カーネル | 優先度 | 難易度 | 状態 | 備考 |
|----------|--------|--------|------|------|
| GEMM (MatMul) | 必須 | 高 | ✅ | 32x32 tiling |
| RMSNorm | 必須 | 低 | ✅ | reduction kernel |
| SiLU | 必須 | 低 | ✅ | element-wise |
| RoPE | 必須 | 中 | ✅ | NTK scaling対応 |
| Softmax | 必須 | 中 | ✅ | numerically stable |
| GQA Attention | 必須 | 高 | ✅ | FlashAttention風fused |
| Embedding | 必須 | 低 | ✅ | gather kernel |
| MoE Router | 必須 | 中 | ✅ | softmax + top-k |
| CrossEntropy | 学習 | 中 | ✅ | forward + backward |
| Aux Loss | 学習 | 中 | ✅ | load balancing |
| AdamW | 学習 | 中 | ✅ | fused optimizer |
| Grad Clip | 学習 | 中 | ✅ | global norm |
| **Decode** | | | | |
| Argmax | 推論 | 低 | ✅ | greedy decoding |
| Sample | 推論 | 中 | ✅ | multinomial + temp |
| TopK Sample | 推論 | 中 | ✅ | top-k sampling |
| TopP Sample | 推論 | 中 | ✅ | nucleus sampling |

---

## Tokenizer / Embedding

### Tokenizer

| 項目 | 値 |
|------|-----|
| 方式 | SentencePiece (自前訓練) |
| アルゴリズム | Unigram or BPE |
| vocab_size | 32000 |
| 特殊トークン | `<pad>`, `<unk>`, `<bos>`, `<eos>` |
| ライセンス | Apache 2.0 |

**訓練データ候補:**
- Wikipedia (日本語 + 英語)
- CC-100 (CommonCrawl)

**訓練コード例:**
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

| 項目 | 値 |
|------|-----|
| vocab_size | 32000 |
| hidden_dim | 2048 |
| パラメータ | 65.5M |
| Weight Tying | なし |
| 初期化 | Normal(0, 0.02) |

### LM Head

| 項目 | 値 |
|------|-----|
| input_dim | 2048 |
| output_dim | 32000 |
| パラメータ | 65.5M |
| bias | なし |

---

## MoE 技術ポイント

1. **Router** — Softmax + Top-K selection
2. **Expert Dispatch** — Token を適切な Expert にルーティング
3. **Expert Combine** — 重み付き出力の集約
4. **Load Balancing Loss** — Expert 利用率の均等化（学習時）
5. **Capacity Factor** — Expert 過負荷時の drop 戦略

---

## NTK RoPE (位置エンコーディング)

### 概要

```
従来の RoPE:
  学習時 context_len を超えると性能劣化

NTK-aware RoPE:
  base frequency をスケールして長コンテキスト対応
  学習なしで context_len を α倍に拡張可能
```

### 設計

| 項目 | 値 |
|------|-----|
| 学習時 context_len | 32K |
| NTK スケール α | 8 |
| 推論時 context_len | **256K** (32K × 8) |
| base frequency | 10000 → 10000 × α^(d/(d-2)) |

### 実装

```python
# NTK RoPE scaling
def ntk_rope_freqs(dim: int, base: float = 10000, alpha: float = 8.0):
    # NTK-aware interpolation
    base = base * alpha ** (dim / (dim - 2))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    return freqs
```

### 利点

1. **学習コスト削減** — 32Kで学習、256Kで推論
2. **追加学習不要** — スケーリングのみで拡張
3. **品質維持** — 長コンテキストでも性能劣化が少ない

---

## 最適化レベル

| Level | 内容 |
|-------|------|
| L1 | Naive CUDA 実装 |
| L2 | Shared memory tiling |
| L3 | FlashAttention, Tensor Core |
| L4 | 量子化 (INT8/INT4) |
