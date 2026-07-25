# MoE Transformer 設計書

## 概要

1.9B Byte-Level Recurrent MoE Transformer の設計仕様。
**BLT + Recurrent (4-loop) + BLT** 構造で超軽量化。
**Shared Expert + Low-Rank Delta** でさらに効率化。
**Julia (Training + GPU) + Rust (Inference/API)** のマルチ言語実装。

---

## 決定事項

- [x] アーキテクチャ: **BLT + Recurrent MoE (8-loop) + BLT**
- [x] 学習: **対応（forward + backward + optimizer）**
- [x] Tokenizer: **Byte-Level (vocab_size=256, 訓練不要)**
- [x] Weight Tying: **しない (Embedding / LM Head 分離)**
- [x] Position Encoding: **RoPE + Timestep Encoding**
- [x] 実装: **Julia (Lux+Reactant) + Rust (Burn) アーキテクチャ確定**
- [x] GPU Decode: **argmax, sample, top-k, top-p 実装完了**
- [x] 型レベル設計: **TensorError, TensorResult 導入**
- [x] Normalization: **Derf (Dynamic erf, 正規化レイヤー不要)**
- [x] MoE 構造: **Shared Expert + Low-Rank Delta (LoRA風, r=256)**
- [x] Optimizer: **Muon (隠れ層) + AdamW (Embedding/LM Head) ハイブリッド**
- [x] Recurrent Stability: **4-loop (実証済み), Timestep Encoding, Early-Exit**
- [x] Precision: **INT4/NF4 (Julia自作実装, ハードウェア非依存)**
- [x] 高速化: **KernelAbstractions.jl + AcceleratedKernels.jl + OhMyThreads.jl + Tullio.jl**

---

## MoE Transformer 仕様

### モデルパラメータ

| パラメータ | Mixtral 8x7B | DeepSeek-MoE | Ours |
|------------|--------------|--------------|------|
| total_params | 46.7B | 16B | **~1.9B** |
| active_params | 12.9B | 2.8B | **~232M** |
| hidden_dim | 4096 | 2048 | **2048** |
| n_layers | 32 | 28 | **BLT×2 + Recurrent×4** |
| n_heads | 32 | 16 | **32** |
| n_kv_heads | 8 (GQA) | 16 | **1 (MQA)** |
| n_experts | 8 | 64 | **128** |
| top_k_experts | 2 | 6 | **8** |
| vocab_size | 32000 | 102400 | **256** |
| context_len | 32768 | 4096 | **4K 訓練** |
| FFN dim/expert | 14336 | 1408 | **16384** |
| head_dim | 128 | 128 | **64** |
| Norm | RMSNorm | RMSNorm | **Derf** |
| Activation | SiLU | SiLU | SiLU |
| Position | RoPE | RoPE | **RoPE** |

### パラメータ計算

```
BLT Input (Byte → Hidden):
  - Embedding:    256 × 2048             =    0.5M
  - Projection:   2048 × 2048            =    4.2M
  BLT Input Total:                       ≈    4.7M

Recurrent MoE Block (4回ループ, 重み共有):
  - Attention:    2048×2048×2 + 2048×64×2 =    8.7M (Q,O + K,V MQA)
  - Router:       2048 × 128               =    0.3M
  - Base Expert:  2048 × 16384 × 3         =  100.7M (shared gate,up,down)
  - Expert Δ:     (2048+16384) × 256 × 3 × 128 = 1811.5M (Low-Rank r=256, 128 experts)
  - Timestep Enc: 4 × 2048                 =    0.008M (loop識別)
  - Derf:         0 (parameterless)        =    0
  Recurrent Block Total:                   ≈ 1921.2M

BLT Output (Hidden → Byte):
  - Projection:   2048 × 2048            =    4.2M
  - LM Head:      2048 × 256             =    0.5M
  BLT Output Total:                      ≈    4.7M

Total: 4.7M + 1921.2M + 4.7M ≈ 1930.6M ≈ 1.9B
Active per token: 4.7M + (8.7M + 100.7M + 113.4M) + 4.7M ≈ 232.2M
```

---

## アーキテクチャ

```
Input Bytes (256 vocab)
    ↓
╔══════════════════════════════════════╗
║         BLT Input Layer              ║
║  Byte Embedding (256 → 768)          ║
║      ↓                               ║
║  Projection (768 → 768)              ║
╚══════════════════════════════════════╝
    ↓
╔══════════════════════════════════════╗
║   Recurrent MoE Block (4 loops)      ║
║   同じ重みを 4回 適用                 ║
║                                      ║
║  Loop 1..4 (timestep encoded):       ║
║    Derf                              ║
║      ↓                               ║
║    MQA Attention + RoPE              ║
║      - Q: 2048 → 2048 (32 heads)     ║
║      - K,V: 2048 → 64 (1 KV head)    ║
║      ↓                               ║
║    + Residual                        ║
║      ↓                               ║
║    Derf                              ║
║      ↓                               ║
║    MoE Layer (128 Experts, top-k=8)  ║
║      Router → [E0..E127] → Mix       ║
║      ↓                               ║
║    + Residual                        ║
╚══════════════════════════════════════╝
    ↓
╔══════════════════════════════════════╗
║        BLT Output Layer              ║
║  Projection (2048 → 2048)            ║
║      ↓                               ║
║  LM Head (2048 → 256)                ║
╚══════════════════════════════════════╝
    ↓
Output Bytes (256 vocab)
```

### Expert FFN (Shared + Low-Rank Delta)

```
Base Expert (共通):
  x → W_gate_base → SiLU ─┐
                          ⊙ → W_down_base → base_out
  x → W_up_base ──────────┘

Expert Delta (選択的, r=256):
  x → (A_gate[i] × B_gate[i]) → SiLU ─┐
                                       ⊙ → (A_down[i] × B_down[i]) → delta_out
  x → (A_up[i] × B_up[i]) ────────────┘

Final Output:
  out = base_out + Σ router[i] × delta_out[i]

Dims: 2048 → 16384 → 2048
Low-Rank: A: 2048×256, B: 256×16384
```

---

## Derf (Dynamic erf) 正規化

### 概要

**Normalization-free** 訓練手法。RMSNorm/LayerNorm を不要にする。

| 項目 | 値 |
|------|-----|
| 正式名称 | Dynamic Error Function |
| 論文 | Stronger Normalization-Free Transformers (2025) |
| 関数型 | Point-wise (要素ごと) |
| パラメータ | なし |

### 核心性質

| 性質 | 効果 |
|------|------|
| Zero-centeredness | 出力を0中心に分布 → Covariate shift 抑制 |
| Boundedness | 有界範囲 [-1, 1] → 訓練安定化 |
| Center Sensitivity | ゼロ付近で敏感 → シグナル伝播 |
| Monotonicity | 単調増加 → 勾配一貫性 |

### 利点

```
RMSNorm:
  - Reduction 必要 (mean, var 計算)
  - スケール param (768 × 2 per layer)
  - メモリ: 統計量保存

Derf:
  - Point-wise のみ
  - パラメータ: 0
  - メモリ: 不要
  - 計算: 高速 (erf は標準関数)
```

### 実装

Julia: `NNlib.erf(x)` または `SpecialFunctions.erf(x)`
Rust: `libm::erf(x)`

---

## Muon Optimizer (ハイブリッド最適化)

### 概要

**Matrix orthogonalization** ベースの optimizer。**隠れ層専用、AdamW と併用必須**。

| 項目 | 値 |
|------|-----|
| 正式名称 | Matrix Orthogonalization Optimizer |
| 適用対象 | 隠れ層の 2D パラメータ (行列のみ) |
| 併用 optimizer | AdamW (Embedding, LM Head, Bias) |
| 効率 | AdamW 比 ~2× (52% FLOPs で同性能) |
| 実績 | Moonlight 3B/16B MoE (5.7T tokens) |

### 役割分担

| レイヤー | Optimizer | 理由 |
|---------|----------|------|
| **Embedding** | AdamW | Sparse updates, 1D parameters |
| **Attention Q/K/V/O** | Muon | 2D matrix, orthogonalization 効果大 |
| **Expert Base/Delta** | Muon | 2D matrix, 隠れ層 |
| **Router** | AdamW | 1D parameters |
| **LM Head** | AdamW | Output layer, sparse targets |
| **Bias** | AdamW | 1D parameters |

### 利点

```
AdamW のみ:
  - 全レイヤー統一
  - 収束遅い (100% FLOPs)
  - メモリ: momentum + variance states

Muon (隠れ層) + AdamW (他):
  - 隠れ層が高速収束 (52% FLOPs)
  - Matrix orthogonalization → 勾配安定化
  - メモリ: やや削減
```

### 実装

Julia: Lux/Optimisers.jl でカスタム実装
Rust: burn-core で optimizer trait 実装

---

## 再帰構造の安定化

### 概要

再帰構造は学習不安定になりがち。2025年研究に基づく安定化戦略。

| 項目 | 値 |
|------|-----|
| ループ回数 | 4回（8回から削減） |
| Timestep Encoding | ループ識別用 |
| Early-Exit | 収束時に動的終了 |
| バッチサイズ | 256K tokens固定 |

### 技術的根拠

**ループ削減 (8 → 4)**:
- 8回ループ: Loss spike + Gradient oscillation
- 4回ループ: 安定性とパフォーマンスのバランス
- 原因: 複数回の再帰反復による勾配フロー複合化

**Timestep Encoding**:
- 各ループ反復にステップ埋め込みを追加
- ループ位置を明示的に識別
- 複雑な反復ソルバーの訓練を容易化
- パラメータ: 4 × 2048 = 8K (negligible)

**Early-Exit (オプション)**:
- Step-norm や二次加速基準で動的終了
- 潜在軌跡の安定化時に自動停止
- 推論時の速度-品質トレードオフ最適化

**バッチサイズ戦略**:
- 256K tokens 固定（1.9B モデル最適）
- 再帰アーキテクチャで安定した勾配推定
- 中規模GPU環境で実現可能

### 参考文献

- [Scaling Latent Reasoning (2025)](https://arxiv.org/html/2510.25741v1) - ループ削減実証
- [Looped Transformers (2025)](https://arxiv.org/html/2410.01405) - Timestep encoding
- [Gradient Flow Matching (2025)](https://arxiv.org/html/2505.20221) - 勾配安定化

---

## Byte-Level Tokenizer (BLT)

### 概要

**Tokenizer 訓練不要**。UTF-8 バイト列を直接処理。

| 項目 | 値 |
|------|-----|
| 方式 | Byte-Level (UTF-8) |
| vocab_size | 256 |
| 特殊トークン | 不要 (バイト列のみ) |
| 訓練 | 不要 |
| 利点 | 全言語・全データ対応 |

### BLT Input Layer

| 項目 | 値 |
|------|-----|
| Byte Embedding | 256 → 2048 (0.5M) |
| Projection | 2048 → 2048 (4.2M) |
| パラメータ | 4.7M |
| 初期化 | Xavier uniform |

### BLT Output Layer

| 項目 | 値 |
|------|-----|
| Projection | 2048 → 2048 (4.2M) |
| LM Head | 2048 → 256 (0.5M) |
| パラメータ | 4.7M |
| bias | なし |

### BLT 動的パッチング戦略

**可変長バイトパッチ**による計算効率化。固定長パッチと異なり、内容に応じて適応的に分割。

| 項目 | 値 |
|------|-----|
| 方式 | Entropy-Based Dynamic Patching |
| 平均パッチサイズ | 4.5-5 bytes/patch |
| FLOP削減 | 最大50% (推論時) |
| 安定化 | EMA Smoothing + STE |

#### 動的パッチング手法（3パラダイム）

| 手法 | メカニズム | 利点 |
|------|----------|------|
| **Entropy-Based (BLT)** | 次バイトエントロピーで境界決定 | 予測困難箇所に計算集中 |
| **Similarity-Based (H-Net)** | 余弦類似度 + EMA スムーシング | 勾配フロー安定 |
| **Compression-Based (ByteFlow)** | 情報理論的コスト評価 | 言語非依存 |

#### 安定化テクニック

| 手法 | 効果 |
|------|------|
| **EMA Smoothing** | 離散決定を連続化 → 勾配伝播可能 |
| **Straight-Through Estimator** | Forward=hard, Backward=soft |
| **Confidence Scoring** | 低信頼度境界を自動補正 |
| **Flash Attention (varlen)** | 可変長シーケンス対応 (6-10× speedup) |

#### 実装アルゴリズム

**Entropy-Driven Segmentation**:
```
Input: byte_seq [b₁, b₂, ..., bₙ]
1. Patcher → entropy H(t) = -Σ p(i)log(p(i))
2. Boundary: H(t) > threshold
3. Patch formation: variable-length groups
Output: patches with 4.5-5 bytes average
```

**Training Stability**:
```
Phase 1: Pretrain Patcher (separate, 1B params)
Phase 2: Freeze Patcher + Train Main Model
Phase 3: (Future) Joint End-to-End Optimization
```

#### 可変長バッチ処理

| 項目 | 実装 |
|------|------|
| Sequence Management | cumsum tensor: `[0, 128, 256, 380, ...]` |
| Attention | `flash_attn_varlen_func` (padding除外) |
| Allocation Strategy | Short → small patches, Long → large patches |

#### 推論最適化

| 最適化 | 効果 |
|-------|------|
| Patch Caching | KVキャッシュ共有 → メモリ 30-40% 削減 |
| Variable-Length Attention | Flash Attention V2 (6-10× speedup) |
| Patcher Quantization | int8化 → 推論3-5% 高速化 |

#### 参考文献

- [Byte Latent Transformer (2025)](https://arxiv.org/abs/2412.09871) - Meta AI, エントロピーベース, 8B scaling
- [H-Net (2025)](https://arxiv.org/abs/2507.07955) - EMA スムーシング, ルーティングモジュール
- [ByteFlow (2025)](https://openreview.net/forum?id=GhJIa921j7) - 情報理論ベース圧縮率
- [MEGABYTE (2023)](https://arxiv.org/abs/2305.07185) - Meta AI, 固定長パッチ先行研究

---

## 位置エンコーディング

### RoPE (Rotary Position Embedding)

**回転位置エンコーディング**。標準的な位置エンコーディング手法。

| 項目 | 値 |
|------|-----|
| 正式名称 | Rotary Position Embedding |
| パラメータ | 0（パラメータレス） |
| メカニズム | 回転行列でクエリとキーに位置情報を付与 |
| 特徴 | 相対位置、外挿性能 |

### 構成

**2層の位置情報**:
1. **トークン位置**: RoPE（シーケンス内の位置）
2. **ループ位置**: Timestep Encoding（1..4のループ識別）

### 利点

| 項目 | 効果 |
|------|------|
| ゼロパラメータ | 追加パラメータ不要 |
| 相対位置 | トークン間の相対的な位置関係 |
| 実績 | LLaMA, GPT-NeoX等で採用 |
| シンプル | 実装が容易、高速 |

### 実装

```
RoPE:
  Q, K に回転行列を適用
  cos/sin による回転で位置エンコード

Timestep Encoding:
  各ループに固有の埋め込み加算
  4 × 2048 = 8K params (negligible)
```

---

## 4-bit量子化 (Julia自作実装)

**Julia + CUDA.jl でゼロから実装**。ハードウェア非依存（A100, RTX 5090等）。

| 項目 | 値 |
|------|-----|
| 手法 | INT4 / NF4 |
| 実装 | Julia (Multiple dispatch + CUDA) |
| 技術 | Hadamard + Per-channel + Stochastic rounding + STE |
| メモリ削減 | 70% (FP16: 3.8GB → INT4: 1.4GB) |

### Mixed-Precision 構成

| コンポーネント | Precision |
|--------------|-----------|
| Embedding, Router, LM Head | BF16 (15%) |
| Attention, MoE Experts | INT4 (85%) |
| Activations, Gradients | BF16 |

### 参考

- [Training Transformers with 4-bit Integers](https://arxiv.org/abs/2306.11987)
- [Optimizing LLM Training Using FP4](https://arxiv.org/abs/2501.17116)
- [Quartet: Native FP4 Training](https://arxiv.org/pdf/2505.14669)

---

## 高速化ライブラリ

### 概要

**CPU + GPU 両対応**の高速化スタック。LoopVectorization.jl は Julia 1.11+ で deprecated のため代替使用。

| レイヤー | ライブラリ | 用途 |
|----------|-----------|------|
| **GPU Kernel** | KernelAbstractions.jl | ベンダー非依存カーネル |
| **GPU Algos** | AcceleratedKernels.jl | sort/reduce/scan (AMD公式) |
| **CPU Parallel** | OhMyThreads.jl | `@tasks` マクロ (LV代替推奨) |
| **CPU Low-level** | Polyester.jl | `@batch` 低オーバーヘッド |
| **Tensor** | Tullio.jl | Einstein notation + 自動GPU |

### 適用箇所

| 対象 | ライブラリ | 効果 |
|------|-----------|------|
| Hadamard変換 | KernelAbstractions | GPU並列化 |
| RoPE計算 | Tullio | Einstein notation + 自動SIMD/GPU |
| Attention | Tullio + AcceleratedKernels | 高スループット |
| Expert FFN | OhMyThreads + Polyester | CPU並列 |
| Softmax/TopK | AcceleratedKernels | GPU reduce/sort |

### KernelAbstractions.jl (GPU Kernel)

| 項目 | 値 |
|------|-----|
| 方式 | `@kernel` マクロ + バックエンド自動選択 |
| NVIDIA | CUDA.jl 経由 |
| AMD | AMDGPU.jl 経由 |
| Apple | Metal.jl 経由 |
| Intel | oneAPI.jl 経由 |
| CPU fallback | Threads.@threads |

### AcceleratedKernels.jl (GPU Algorithms)

| 項目 | 値 |
|------|-----|
| 対応アルゴリズム | sort, reduce, accumulate, map, foreachindex |
| 実績 | AMD公式ソートに採用 |
| オーバーヘッド | <7% |
| 利点 | 単一APIで全GPU対応 |

### OhMyThreads.jl (CPU Parallel)

| 項目 | 値 |
|------|-----|
| マクロ | `@tasks`, `@set` |
| 特徴 | LoopVectorization.jl 代替推奨 |
| スケジューラ | work-stealing |
| 利点 | モダンAPI、型安定 |

### Tullio.jl (Tensor Operations)

| 項目 | 値 |
|------|-----|
| 記法 | Einstein notation (`@tullio C[i,j] := A[i,k] * B[k,j]`) |
| CPU | 自動SIMD (LoopVectorization経由 or fallback) |
| GPU | KernelAbstractions経由で自動生成 |
| 利点 | 可読性 + 高性能 |

### 使用例

```julia
using Tullio, KernelAbstractions, AcceleratedKernels, OhMyThreads

# Attention (Tullio)
@tullio scores[b,h,i,j] := q[b,h,i,d] * k[b,h,j,d]

# Softmax (AcceleratedKernels)
softmax_out = AK.softmax(scores; dims=4)

# Expert FFN (OhMyThreads)
@tasks for i in 1:n_experts
    expert_out[i] = ffn(x, expert_weights[i])
end
```

### 参考

- [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/dev/)
- [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl)
- [OhMyThreads.jl](https://github.com/JuliaFolds2/OhMyThreads.jl)
- [Tullio.jl](https://github.com/mcabbott/Tullio.jl)

---

## プロファイリング

### 概要

**BenchmarkTools.jl** でパフォーマンス計測。標準 `@code_warntype` で型安定性検証。

| 項目 | 値 |
|------|-----|
| ベンチマーク | BenchmarkTools.jl (`@btime`, `@benchmark`) |
| 型検査 | 標準 `@code_warntype` |
| デバッガ | VSCode Julia extension |

### 推奨ワークフロー

```
1. 開発フェーズ
   └─ Revise.jl (自動リロード)
   └─ VSCode debugger (ブレークポイント)

2. 型安定性検証
   └─ @code_warntype (標準、追加依存なし)

3. パフォーマンス最適化
   └─ BenchmarkTools.@btime (計測)
```

### 使用例

```julia
using BenchmarkTools

# パフォーマンス計測
@btime forward($model, $x)

# 型安定性検証（標準機能）
@code_warntype forward(model, x)
```

### 参考

- [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
