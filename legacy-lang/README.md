# legacy-lang

`machine_learning` 時代（〜2026-01-18、commit `c02bbea`）の多言語実装。2026-02-10 の `2c0c371 Rewrite SLM benchmark: 4-language MoE Transformer with unified BLAS` で現行構成（`rust/` `julia/` `python/` `go/` の各言語独立実装）へ全面的に置き換えられたため、本体からは外れている。

2026-07-26 に、旧クローン `GitHub/learning/` を廃止する際にここへ移送した。**移送元ではコミットされておらず（index にstageされたまま放置）、リポジトリ履歴のどこにも存在しない**ため、これが唯一の実体。

## なぜ残すか

現行構成に**言語間FFIが存在しない**ため。`ffi_julia` は Rust から Julia の `MLCore.jl` を直接駆動する唯一の実装で、jlrs によるゼロコピー連携・型安全ラッパ・config駆動を持つ。現行はベンチマークのために各言語で同一アルゴリズムを独立実装する方針なので、FFI経路は後継がない。

## 構成（25ファイル・2,827行）

| パス | 内容 |
|---|---|
| `rust/ffi_julia/` | Rust ⟷ Julia FFI（jlrs）。`src/lib.rs` 304行、`examples/{basic,inference,training}.rs`、`tests/integration.rs`、`README.md` |
| `rust/api/` `rust/cli/` `rust/inference/` `rust/web/` | 各エントリポイントのcrate雛形 |
| `julia/src/MLCore.jl` | BLT Recurrent MoE Transformer 本体、534行 |
| `julia/{Project,Manifest}.toml` `julia/config.yaml` | Julia側の依存とモデル設定 |
| `julia/test/` `julia/test_model.jl` | テスト3本 |

`ffi_julia` が謳う機能: config駆動のモデル設定読込、Rust構造体とJulia型の対応、entropy-based dynamic patching、CPU/GPU双方の学習・推論、builderパターン。

## 注意

- **ビルド検証していない**。移送時点で `lang/rust/target/` は破棄しており、`jlrs` の依存解決やJulia側のバージョン整合は未確認
- 現行の `rust/` 配下とは Cargo workspace が独立している。統合するなら依存とエディションの再調整が必要
- 設計の背景は [`docs/legacy/rationale.md`](../docs/legacy/rationale.md)、当時のロードマップは [`docs/legacy/roadmap.md`](../docs/legacy/roadmap.md)
