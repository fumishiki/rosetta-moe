<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->

# Loss Convergence Revision History (Detailed Engineering Log)

## Scope

This document records the trial-and-error process behind the loss convergence pipeline updates made on **2026-02-12**.

It exists for two reasons:

1. Preserve the debugging context (what failed, why, and how it was fixed).
2. Make future edits safer by documenting hidden constraints (sandbox, reproducibility, parser assumptions).

## Baseline Problem Statement

We wanted a per-language convergence workflow that:

- runs each implementation one-by-one (Rust, Go, Python, Julia),
- saves raw loss sequences as JSON,
- renders clean per-language graphs,
- and keeps README values synced with generated artifacts.

## Chronological Log

### 1) Initial implementation

- Added `scripts/convergence_plots.py`.
- Added `make convergence-plots` in `Makefile`.
- Added SVG embedding and artifact paths in README.

Initial design assumptions:

- each language command prints one JSON payload,
- command execution is deterministic enough to keep README values stable,
- Go test execution can run in sandbox with default cache paths.

### 2) First failure: Go cache permission error in sandbox

Failure observed while running the new script:

- Go failed with `operation not permitted` for:
  - `~/Library/Caches/go-build/...`
  - `~/Library/Caches/go-build/trim.txt`

Root cause:

- sandbox restrictions prevented writes to default Go cache locations.

Fix attempt:

- script updated to override `GOCACHE` and `GOMODCACHE` to local writable paths.

Final decision:

- moved Go caches to `/tmp/rosetta-moe-go-build-cache` and `/tmp/rosetta-moe-go-mod-cache`.
- avoids polluting the repo and remains writable in the current execution environment.

### 3) First reproducibility concern: Go numbers shifted across runs

Observation:

- Rust/Python/Julia gave stable initial/final loss values across reruns.
- Go initial/final values changed between runs.

Impact:

- README table could drift after every regeneration.
- graph and table consistency was hard to maintain.

Investigation:

- Go convergence path uses `go test -run TestConvergence`.
- Model initialization uses `math/rand`-based normal sampling.
- Convergence test did not explicitly set RNG seed.

Root cause:

- missing `rand.Seed(...)` in Go `TestConvergence`.

Fix:

- imported `math/rand` in `go/nn_test.go`.
- added `rand.Seed(42)` at the beginning of `TestConvergence`.

Verification:

- reran Go convergence twice.
- both runs produced identical values (`7.9454 -> 0.0247`).

### 4) Parser and automation friction during validation

Issue:

- ad hoc command-line extraction logic failed multiple times due strict regex assumptions.
- one run failed because helper command referenced an undefined shell variable in inline Python formatting.

Fix:

- standardized extraction to parse the last valid JSON object line.
- reduced dependency on fragile regex where possible.

Lesson:

- convergence automation should never depend on exact log line ordering.

### 5) Synchronization pass (artifacts and docs)

After reproducibility fix:

- reran `make convergence-plots`.
- regenerated:
  - `benchmarks/convergence/{rust,go,python,julia}.json`
  - `docs/assets/convergence/{rust,go,python,julia}.svg`
- updated README convergence table to match latest generated values.

### 6) Python optimization pass (Rust-inspired hot-path cleanup)

After convergence became stable, we focused on Python runtime overhead.

Observed bottleneck before optimization:

- `train_step` average was **6.1882 ms/step** (tiny model, batch=2, seq=8).
- profiler showed `Trainer._adamw_step` dominating runtime.
- secondary hotspots were MoE dispatch (`moe.py`) and attention softmax/temp arrays.

#### 6.1 First optimization wave (`train.py`)

Changes:

- moved AdamW updates to in-place operations with reusable scratch buffers
- computed LR / bias-correction constants once per step (not per parameter)
- switched gradient clipping to in-place scaling instead of creating scaled copies
- removed `probs.copy()` in cross-entropy gradient path and reused softmax buffer
- cached `model.parameters()` once in `Trainer` instead of rebuilding each step

Result:

- `train_step` average improved from **6.1882 ms** to **3.0842 ms** (~50.2% faster).

#### 6.2 Second optimization wave (`moe.py`, `layers.py`, `attention.py`)

The next bottleneck was no longer optimizer math but dispatch/allocation patterns.

Changes:

- **MoE routing path**:
  - replaced repeated per-expert boolean-mask scans with token->expert inversion + grouped dispatch
  - removed unnecessary backward input reconstruction and reused forward-side caches
  - reduced reliance on `np.add.at` where index uniqueness was guaranteed
- **Router softmax**:
  - added reusable softmax buffer (`_probs_buf`) for gate probabilities
- **Linear/SwiGLU path**:
  - removed avoidable dtype-copy conversions in backward accumulation
  - reduced temporary Tensor allocation in SwiGLU cache handling
- **Attention path**:
  - cached RoPE trig tensors (`cos/sin`) per `seq_len`
  - switched attention softmax to in-place transformations

Result:

- `train_step` average improved further from **3.0842 ms** to **2.8829 ms**.
- additional gain: **~6.53%** over wave 1.
- total gain vs original baseline: **~53.41%**.

#### 6.3 Safety checks after optimization

To ensure speed changes did not silently break correctness:

- Python tests: `31 passed`
- convergence re-run: unchanged (`7.3422 -> 0.0233` over 1000 steps)

Key takeaway:

- the hardest part was not “finding one magic optimization,” but preserving correctness
  while removing temporary allocations and redundant work across multiple files.
- each wave shifted the bottleneck; profiling had to be repeated after every significant change.

### 7) Go optimization pass (SwiGLU clone reduction + broad hot-path audit)

After Python, we ran a dedicated Go pass focused on allocation pressure in training.

#### 7.1 User-requested target: `layers.go` `SwiGLU` clone reduction

Starting point:

- `SwiGLU.Forward` cached both pre-SiLU and post-SiLU tensors using `Clone()`.
- backward used the cached post-SiLU tensor only to compute `gradUp`.

Iteration:

- removed the post-SiLU clone first and recomputed `silu(pre_silu)` during backward.
- then removed the remaining pre-SiLU tensor clone by storing pre-SiLU values in a reusable `[]float32` buffer.

Final approach:

- `lastGatePreSiLU` changed from `*Tensor` clone to reusable raw slice.
- backward computes both:
  - `gradUp = gradHidden * silu(pre_silu)`
  - `gradSiluGate *= silu'(pre_silu)`
- gradient merge uses in-place accumulation to avoid an extra temporary tensor.

Why this mattered:

- this path executes for every expert FFN call and was multiplying copy cost under MoE routing.

#### 7.2 First failed optimization attempt in `train.go`

Attempt:

- pre-scaled all gradients once when clipping triggered, to avoid `* clipCoeff` in the inner Adam loop.

Observed regression:

- `mem_train_step` latency spiked in one run (median moved to ~4.19 ms range),
  because the additional full gradient pass increased memory traffic.

Fix:

- reverted to coefficient application inside Adam update.
- added a fast branch: if `clipCoeff == 1.0`, skip the multiply entirely.

Lesson:

- removing a multiply is not always a win if it introduces another full-array pass.

#### 7.3 Additional broad Go review fixes (beyond SwiGLU)

To address remaining weak spots after the SwiGLU change:

- `attention.go`:
  - removed unnecessary `Q/K/V` cloning in forward (safe by lifetime in current graph flow),
  - switched gradient sum to in-place add,
  - added reusable backward buffers (`gradQ/gradK/gradV/gradScores`).
- `moe.go`:
  - reused router aux-loss buffers,
  - reused expert-token and weight-index group buffers,
  - reused expert batch/grad buffers and backward input-grad buffer.

These were not speculative changes; each was selected because it sat on the train-step hot path.

#### 7.4 Validation and noise handling

Correctness checks after each wave:

- Go tests: pass (`go test -run 'Test[^B]' -count=1 ./...`)
- convergence: unchanged behavior (`7.9454 -> 0.0247` over 1000 steps)

Benchmark challenge:

- single-run timing was noisy, with occasional outliers.
- we switched to repeated runs and tracked trend + allocation deltas instead of trusting one datapoint.

Observed stable effect (this optimization round):

- `mem_train_step` allocations dropped from about `26.32 MB` to about `25.49 MB` per scenario run.
- `parallel_train_T4` allocations dropped from about `105.09 MB` to about `101.82 MB`.
- latency improved on several runs, but with notable variance; allocation reduction was the more reliable signal.

Key takeaway:

- the real progress came from cumulative allocation cleanup across SwiGLU, attention, and MoE,
  not from one isolated micro-optimization.
- for Go in this project, "measure repeatedly" was mandatory; one benchmark run was not trustworthy.

## Final Stable State (as of 2026-02-12)

- `make convergence-plots` runs all 4 languages sequentially.
- JSON and SVG artifacts are regenerated in deterministic locations.
- Go convergence is reproducible across repeated runs.
- Python training hot path is significantly optimized while preserving convergence behavior.
- README references generated files and current values.

## Why this log matters

Without this log, a future maintainer will likely repeat the same mistakes:

- assuming Go cache writes are always available in sandbox,
- assuming Go convergence output is deterministic without explicit seeding,
- assuming a strict regex parser is robust enough for toolchain output changes.

This file is intentionally detailed so the reasoning and pain points are not lost.

### 8) Rust final optimization sweep ("apply all remaining hot-path fixes")

After the earlier Rust pass, we still had two suspicious hotspots:

1. attention cache cloning (`MQAttention`),
2. repeated routing/grouping allocations in `MoE`.

#### 8.1 `attention.rs`: removed redundant cache clones

What was wasteful:

- `MQAttention::forward` cached input via `to_vec()` even though sub-projections (`q_proj/k_proj/v_proj`) already cache their own inputs.
- `Q/K/V` were cached with `to_vec()` even though ownership could be moved after the attention compute loop.

Fix:

- removed duplicate `last_input`/`last_input_shape` cache from `MQAttention`.
- switched Q/K/V cache to ownership move (`into_data`) after forward compute.
- removed backward-time input cache reconstruction for `q_proj/k_proj/v_proj`.

Result:

- less copy pressure in training backward,
- no behavior change in tests (`cargo test` still passes).

#### 8.2 `moe.rs`: reused route index and expert-group buffers

What was wasteful:

- per-step `Vec<Vec<usize>>` recreation for top-k route indices,
- per-step expert token grouping vectors recreated and dropped.

Fix:

- Router now reuses `indices_buf` in addition to `probs_buf`.
- `MoELayer` now keeps persistent expert grouping buffers:
  - `expert_tokens_buf`
  - `expert_weight_idx_buf`
- route indices/probabilities are returned to Router caches in backward (`take()` + move-back pattern), same style as the probability buffer optimization.

#### 8.3 Validation and noisy benchmark interpretation

Why this was tricky:

- one benchmark run showed severe latency regression, but repeated runs showed large jitter.
- this project's tiny scenarios are sensitive to background load / scheduler noise.

Verification strategy:

- required repeated runs and allocation-focused comparison (not single-run latency).
- re-ran `cargo fmt`, `cargo test`, `cargo run --release --bin bench`, and convergence.

Observed stable effects after this sweep:

- `mem_train_step`: allocation dropped from ~`1,393,700` to ~`983,787` bytes.
- `parallel_train_T4`: allocation dropped from ~`5,574,800` to ~`3,935,148` bytes.
- `scale_train_256`: allocation dropped from ~`5,104,804` to ~`3,721,553` bytes.
- latency generally improved or stayed close to baseline depending on run variance.

Takeaway:

- for this Rust codebase, buffer lifecycle design (move/reuse) gave more reliable gains than micro-tuning arithmetic loops.
- single-shot timing was misleading; repeated measurements were mandatory.

### 9) Julia complementary optimization (loss+grad fusion in CE path)

To keep "all optimization" consistent across languages, we added one high-confidence Julia hot-path optimization.

#### 9.1 `train.jl`: fused cross-entropy loss and gradient

Before:

- `train_step!` ran:
  - `cross_entropy_loss(...)` (softmax + permute pass),
  - then `cross_entropy_grad_into!(...)` (another softmax + permute pass).

After:

- added `cross_entropy_loss_grad_into!`:
  - computes loss and logits gradient in one fused pass,
  - reuses existing `perm_buf`, `grad_perm_buf`, and `grad_buf`,
  - removes one full softmax/permutation traversal per step.

Validation:

- Julia tests: `71/71` pass.
- benchmark run confirmed expected training-step level performance (`mem_train_step` around `0.94 ms` in this environment).
- convergence script still runs successfully with valid loss output.

Takeaway:

- Julia optimization wins were strongest when removing duplicated full-array passes, not by adding lower-level loop tricks.

### 10) Julia re-inspection pass (MoE buffer-copy cleanup)

After the fused CE update, we re-inspected Julia again and found another practical hotspot in `moe.jl`.

#### 10.1 Problem observed

`MoELayer` used grow-only expert buffers and then sliced `bb[1:n_tok, :]` when token counts were smaller than capacity.
This introduced repeated row-slice copy paths in both forward and backward.

#### 10.2 Fix applied

- switched expert forward buffers to exact-shape per-expert matrices (no slice fallback copy),
- replaced shared backward scratch matrices with per-expert backward buffers:
  - `bwd_expert_grad_bufs`
  - `bwd_expert_input_bufs`
- replaced broadcast row updates with explicit SIMD-friendly loops in gather/scatter paths.

#### 10.3 Verification

- Julia tests still pass (`71/71`).
- convergence output remains stable (same loss curve behavior).
- benchmark improved versus the immediate pre-fix recheck run:
  - `mem_train_step`: `1,108,021ns -> 1,037,750ns`, alloc `2,992,448 -> 2,740,608`
  - `parallel_train_T4`: `5,131,083ns -> 4,706,500ns`, alloc `11,863,968 -> 10,899,168`
  - `scale_train_256`: `11,164,437ns -> 9,910,708ns`, alloc `9,973,888 -> 9,120,128`

Takeaway:

- the remaining Julia gains came from removing hidden copy paths in MoE batch plumbing,
  not from changing math kernels.
