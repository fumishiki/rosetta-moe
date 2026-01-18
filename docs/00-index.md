# cuda-nn ドキュメント

## 概要

MoE Transformer (6.9B total / 1.8B active) のマルチ言語実装。
Rust + Go + Python + CUDA でフルスクラッチ実装。

---

## ドキュメント一覧

| ドキュメント | 内容 |
|-------------|------|
| [1-model.md](1-model.md) | モデルアーキテクチャ設計 |
| [2-learn.md](2-learn.md) | 学習システム設計 |

---

## プロジェクト構成

```
machine_learning/
├── crates/
│   ├── cuda/           # 共有CUDAカーネル (9ファイル)
│   │   ├── kernels/    # .cu カーネルファイル
│   │   └── src/        # stub.c (CPU fallback)
│   ├── rust/           # Rust実装
│   │   ├── nn-core/    # モデル・テンソル・学習
│   │   └── nn-ffi/     # CUDA FFIブリッジ
│   ├── go/             # Go実装
│   │   ├── tensor/     # テンソル操作
│   │   ├── cuda/       # cgo CUDAバインディング
│   │   ├── layer/      # NN層
│   │   ├── model/      # MoEモデル
│   │   └── train/      # 学習パイプライン
│   └── python/         # Python実装
│       ├── nn/         # NNモジュール
│       ├── cuda/       # ctypes CUDAバインディング
│       └── tests/      # pytest テスト
└── docs/               # 設計ドキュメント
```

---

## 実装言語比較

| 項目 | Rust | Go | Python |
|------|------|-----|--------|
| テンソル | 独自型 + Error型 | 独自型 | numpy backend |
| CUDAバインディング | FFI (build.rs) | cgo (Makefile) | ctypes |
| CPU fallback | stub.c | stub.c | numpy |
| テスト数 | 53 | 31 | 42 |
| 高度な最適化 | ✅ (CUDA Graph等) | - | - |

---

## クイックスタート

### Rust

```bash
cargo build --release
cargo test
```

### Go

```bash
cd crates/go
go test ./...
```

### Python

```bash
cd crates/python
pip install -e ".[dev]"
pytest
```

---

## モデル仕様

| パラメータ | 値 |
|-----------|-----|
| 総パラメータ | ~6.9B |
| アクティブパラメータ | ~1.8B |
| Hidden dim | 768 |
| Layers | 30 |
| Attention | MQA (12Q/1KV) |
| Experts | 16 total, top-4 active |
| FFN dim | 6144 |
| Vocab size | 32,000 |
| Context | 32K train → 256K inference (NTK RoPE) |

---

## 主要コンポーネント

### モデル層

- **Embedding**: トークン埋め込み (32K × 768)
- **RMSNorm**: Root Mean Square正規化
- **MQA Attention**: Multi-Query Attention (12Q/1KV)
- **MoE Layer**: Router + 16 Experts (top-4選択)
- **SwiGLU FFN**: Gated Linear Unit (768 → 6144 → 768)
- **LM Head**: 出力投影 (768 → 32K)

### CUDA カーネル

| ファイル | カーネル |
|----------|----------|
| elementwise.cu | silu, add, mul, scale |
| softmax.cu | softmax, softmax_topk |
| rmsnorm.cu | rmsnorm, rmsnorm_residual |
| gemm.cu | gemm, gemm_batched |
| rope.cu | rope_freqs, rope_forward |
| attention.cu | attention_scores, flash_attention |
| loss.cu | cross_entropy, aux_loss |
| optimizer.cu | adamw_step, grad_clip, scatter_add |
| decode.cu | argmax, sample, topk_sample, topp_sample |

### 学習機能

- **Loss**: CrossEntropy + MoE AuxLoss (load balancing)
- **Optimizer**: AdamW (β1=0.9, β2=0.95)
- **LR Schedule**: Warmup + Cosine Decay
- **Decode**: Greedy, Sample, Top-K, Top-P

---

## テスト状況

| 言語 | テスト数 | 状態 |
|------|----------|------|
| Rust | 53 | ✅ |
| Go | 31 | ✅ |
| Python | 42 | ✅ |
| **総計** | **126** | ✅ |

---

## ライセンス

MIT License
