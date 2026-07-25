# docs/legacy

`machine_learning` 時代の設計文書4本。2026-07-26 に旧クローン `GitHub/learning/` から移送した。

移送元では `c02bbea`（2026-01-18）にコミット済みだったが、リポジトリ側では `6e275f7 Remove Japanese docs` で `docs-jp/` `docs-en/` ごと削除されている。履歴に残る `docs-jp/1-model.md` は251行で、**ここにある版（580行）とは741行の差**がある。つまり履歴版は同一文書の古い断面であり、こちらが最新版。

| ファイル | 行数 | 内容 |
|---|---:|---|
| `1-model.md` | 580 | MoE Transformer 設計書 |
| `roadmap.md` | 518 | 実装ロードマップ。2026-02-03時点、Phase 0→6（30–40時間想定） |
| `architecture.md` | 439 | 多言語アーキテクチャ仕様 |
| `rationale.md` | 156 | 多言語構成を採る理由と言語間比較 |

現行の設計は [`docs/spec.md`](../spec.md)、ベンチ結果は [`docs/bench-results.md`](../bench-results.md) にある。ここの4本は**現行構成（2026-02-10 の全面書き直し以降）を説明しない**ので、実装の参照元にはしないこと。設計判断の経緯を追うためだけに残している。

対応する実装は [`legacy-lang/`](../../legacy-lang/README.md)。
