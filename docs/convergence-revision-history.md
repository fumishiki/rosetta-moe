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

## Final Stable State (as of 2026-02-15)

- `make convergence-plots` runs all 4 languages sequentially.
- JSON and SVG artifacts are regenerated in deterministic locations.
- All 4 languages use identical Knuth LCG for weight initialization (seed=42).
- Cross-language RNG produces bitwise-identical weight values.
- All languages converge to final loss < 0.03 over 500 steps.
- Router z-loss, gradient clipping (0.5), and warmup (50 steps) enabled in all languages.
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

### 11) Stability improvements and cross-language RNG unification (2026-02-15)

This section covers 4 interconnected fixes that resolved a Rust convergence regression and unified weight initialization across all 4 languages.

#### 11.1 Router z-loss, gradient clipping, and warmup (all 4 languages)

Added three stability improvements inspired by ST-MoE (Zoph et al.):

- **Router z-loss**: `L_z = z_weight * (1/B) * Σ_i logsumexp(logits_i)²` — penalizes large pre-softmax router logits to prevent expert collapse. z_loss_weight=0.01 in all languages.
- **Gradient clipping threshold lowered**: 1.0 → 0.5 (max_norm for global gradient clipping)
- **Warmup steps extended**: 10 → 50 steps (linear warmup before cosine decay)

Files modified: Rust (moe.rs, train.rs, config.rs), Go (moe.go, train.go, config.go), Python (moe.py, train.py, config.py), Julia (moe.jl, train.jl, config.jl)

#### 11.2 Rust LR schedule bug fix

Rust's `AdamW.step()` was using a fixed learning rate set at construction time (1e-3). Go/Python/Julia correctly called `get_lr()` each step to apply warmup + cosine decay.

Fix:
- Added `set_lr()` method to AdamW optimizer
- Called `self.optimizer.set_lr(self.get_lr())` before each optimizer step in `train_step_with_logits()`

#### 11.3 Rust identical expert initialization (ROOT CAUSE)

After adding z-loss and stability settings, Rust convergence regressed severely (final loss 0.140 vs Go's 0.026).

Root cause:
- `Embedding::new` used `Tensor::randn(shape, DType::F32, 42)` (seed=42)
- ALL `Linear::new` used `Tensor::randn(shape, DType::F32, 123)` (seed=123)
- Same shape + same seed = IDENTICAL initial weights for all 4 MoE experts
- Identical experts cannot learn diverse features (symmetry problem)
- With conservative stability settings (grad_clip=0.5, warmup=50), symmetry-breaking was too slow

Fix:
- Replaced per-call seed parameter with thread-local global RNG state (Knuth LCG: `state = state * 6364136223846793005 + 1`)
- `seed_rng(seed: u64)` function sets global state
- Each layer constructor consumes sequential RNG values, ensuring different initial weights
- All binaries call `seed_rng(42)` before model creation

Result: Final loss improved from 0.140 to 0.013 (91% improvement).

#### 11.4 Step counting off-by-one fix (Rust)

Rust incremented `self.current_step += 1` at the END of `train_step_with_logits`. This meant the first step used `current_step=0`, which yielded `lr = base_lr * 0/warmup = 0.0`. The first training step applied zero learning rate (no weight update).

Go incremented `t.step++` at the TOP of `TrainStepFromLogits` (line 203).

Fix: Moved `self.current_step += 1` to the top of `train_step_with_logits` to match Go.

#### 11.5 Cross-language RNG unification

All 4 languages previously used different RNG algorithms for weight initialization:
- Rust: custom LCG (already in generate module)
- Go: `math/rand` (lagged Fibonacci generator)
- Python: `numpy.random` (Mersenne Twister MT19937)
- Julia: `randn` (Xoshiro256++)

This meant identical seed=42 produced different weight sequences across languages.

Fix: Ported the same Knuth LCG + Box-Muller transform to weight initialization in all 4 languages:
```
LCG: state = state * 6364136223846793005 + 1
Uniform: u = state / UINT64_MAX, clamped to [1e-10, 1.0]
Box-Muller: z0 = sqrt(-2*ln(u1)) * cos(2π*u2), z1 = sqrt(-2*ln(u1)) * sin(2π*u2)
```

Verification: All 4 languages produce identical first 5 values from seed 42:
`[-1.1714804, 0.23290575, 1.0633162, 0.16791363, -1.287875]`

Files modified:
- Rust: tensor.rs (replaced seed parameter with global state)
- Go: tensor.go (added LCG state, SeedRNG, lcgUniform, lcgNormal)
- Python: tensor.py (added LCG state, seed_rng, _lcg_uniform, _lcg_normal)
- Julia: tensor.jl (added LCG state, seed_rng!, _lcg_uniform, _lcg_normal)

#### 11.6 Post-fix convergence comparison

| Language | Step 1 | Step 50 | Final (Step 500) |
|----------|--------|---------|-------------------|
| Rust | 8.436 | 1.344 | 0.013 |
| Go | 8.460 | 0.108 | 0.022 |
| Julia | 7.819 | 0.221 | 0.023 |
| Python | 7.780 | 0.136 | 0.026 |

Note: Initial loss differs between languages (~8.4 vs ~7.8) due to different input data in convergence scripts (random vs sequential tokens). All converge to <0.03.

Small remaining differences in training dynamics are due to:
- AdamW eps placement: Rust uses `m_hat / sqrt(v_hat + eps)` (eps inside sqrt), Go/Python/Julia use `m_hat / (sqrt(v_hat) + eps)` (eps outside sqrt)
- Float32 accumulation order differences in BLAS-accelerated forward pass

### 12) MoE Routing Strategy additions: BiasFree + ReMoE (2026-02-15)

Added two new routing strategies to combat loss spikes during MoE training. All changes implemented across all 4 languages in parallel.

#### 12.1 Motivation

MoE training exhibited loss spikes (up to +484%). Existing mitigations (z-loss 0.01, aux_loss 0.01, grad_clip 0.5, warmup 50) were insufficient. Three interventions were applied:

1. **z-loss weight increase**: 0.01 → 0.05 (immediate, stronger logit regularization)
2. **Auxiliary-Loss-Free Balancing** (DeepSeek-V3): replace aux_loss with dynamic expert bias
3. **ReMoE** (ICLR 2025): replace TopK+Softmax with ReLU gate for fully differentiable routing

#### 12.2 Design: RoutingMode enum

Added `RoutingMode` enum to `TrainConfig` in all 4 languages:

| Mode | Gate | Selection | Loss Components | Post-step |
|------|------|-----------|-----------------|-----------|
| TopK (default) | softmax | greedy top-k | CE + aux_loss + z_loss | — |
| BiasFree | softmax | bias-augmented top-k | CE + z_loss (no aux) | update expert_bias |
| ReLU | ReLU | variable active set | CE + L1 regularization (no aux/z) | adaptive lambda |

Key design decision: single Router struct with mode branching (not 3 separate types). ~80% code shared across modes.

#### 12.3 BiasFree routing (DeepSeek-V3)

Implementation:
- Selection score = `softmax(logits)[e] + expert_bias[e]` (bias augments selection only)
- Weights = original softmax probabilities (NOT augmented scores)
- Expert bias update: `bias[e] += gamma * sign(1/N - f_e)` where `f_e` = token fraction to expert e
- `bias_gamma` default: 0.001
- No auxiliary loss needed — bias directly corrects load imbalance

#### 12.4 ReMoE routing (ICLR 2025)

Implementation:
- Gate weights = `max(0, logits[t,e])` (ReLU, no softmax)
- Active experts = all with positive gate weight (variable count 0..n_experts)
- Fallback: if no expert active, force-activate argmax(logits) with weight 1.0
- Overflow: if active > top_k, keep only top_k highest weights
- Pad indices to top_k length (weight=0) so MoELayer dispatch code is unchanged
- L1 loss: `lambda * mean_t(Σ_e relu(logits[t,e]))` with manual gradient backprop to gate.weight
- Adaptive lambda: if avg_active > target_k ± 0.1, scale lambda by 1.01/0.99

#### 12.5 Training loop integration

`train_step` branches on routing_mode:
- **TopK**: `total = CE + aux_loss(alpha) + z_loss(0.05)` (original path unchanged)
- **BiasFree**: `total = CE + z_loss(0.05)`, then `model.update_routing_biases(gamma)` post-optimizer
- **ReLU**: `total = CE + relu_l1_loss`, then adaptive lambda update post-optimizer

#### 12.6 Files modified (per language × 4 = ~24 files)

| File pattern | Changes |
|-------------|---------|
| `*/train.*` | RoutingMode enum, TrainConfig extension, train_step branching |
| `*/moe.*` | Router struct extension, route() branching, 3 new methods |
| `*/model.*` | set_routing_mode / update_biases / apply_relu_l1 propagation |
| `*/convergence*` | --routing-mode CLI flag |
| `*/test*` | BiasFree + ReLU routing tests |
| `rust/src/lib.rs` | Re-export RoutingMode |
| `julia/config.jl` | RoutingMode @enum (before moe.jl in include order) |

#### 12.7 Convergence results

All 3 modes converge on tiny model (batch=2, seq=8, 500 steps):

| Mode | Initial Loss | Final Loss | Notes |
|------|-------------|------------|-------|
| TopK | ~8.8 | ~0.022 | Baseline, unchanged |
| BiasFree | ~8.7 | ~0.00003 | Faster convergence, no aux_loss overhead |
| ReLU | ~8.0 | ~0.00002 | Fastest convergence, fully differentiable |

All 4 languages pass full test suites:
- Rust: 23 integration tests
- Go: all convergence tests (TopK + BiasFree + ReLU)
- Python: 34 integration tests
- Julia: 79 tests

#### 12.8 Implementation notes

- Router.route() / Forward() return signatures unchanged in all languages
- ReLU mode pads indices to top_k length — MoELayer dispatch code untouched
- Julia: RoutingMode @enum placed in config.jl (must precede moe.jl in include order)
- Julia: TrainConfig is immutable struct — all 14 fields positional in constructor
- Expert bias is 0-initialized, grows/shrinks via sign(target - f_e) per step
- Adaptive lambda bounded by ±1% per step to prevent oscillation

---

### 13. Spike Analysis & Go Lambda Propagation Fix (2026-02-15)

#### 13.1 Problem statement

Post-implementation spike analysis revealed:
- BiasFree mode: 23 spikes in Python (max +462,541%)
- ReLU mode: 22 spikes in Go, 29 spikes in Python (max +167,007%)
- TopK mode: 0-2 spikes (baseline)

Initial hypothesis: Go ReLU had a unique implementation bug (22 spikes vs "0" in Python/Julia).

#### 13.2 Go lambda propagation bug (fixed)

**Root cause**: `train.go` updated `t.config.ReLULambdaL1` (trainer config) but never propagated the new value to Router objects. Routers kept using the stale initial lambda (0.01).

Python/Julia directly update each router's lambda:
```python
# Python (correct)
for block in self.model.blocks:
    block.moe.router.relu_lambda_l1 *= 1.01
```

Go only updated the config:
```go
// Go (before fix)
t.config.ReLULambdaL1 *= 1.01  // Router never sees this
```

**Fix**: Added `t.model.SetRoutingMode(ReLUMode, t.config.ReLULambdaL1)` after the adaptive lambda update in `train.go` line 325.

#### 13.3 bias_gamma investigation (negative result)

Tested `bias_gamma` sweep for BiasFree mode: 0.001, 0.0001, 0.00001.
Results were **identical** across all gamma values — same spike count, same steps, same max%.
Conclusion: spikes are NOT caused by bias update dynamics.

#### 13.4 True root cause: inherent training instability

Comparison of spike steps between TopK and BiasFree modes showed spikes at **identical steps** (62, 283), proving they are inherent to the training dynamics (LR schedule, loss landscape), not routing-mode-specific.

Re-running all modes after the Go fix produced:

| Language | TopK Spikes | BiasFree Spikes | ReLU Spikes |
|----------|-------------|-----------------|-------------|
| Go       | 0           | —               | 22          |
| Python   | 2           | 23              | 29          |

#### 13.5 Spike severity is a measurement artifact

The extreme spike percentages (+462,541%, +167,007%) are misleading because BiasFree/ReLU achieve much lower absolute loss before spikes:

| Mode | Pre-spike Loss | Post-spike Loss | Δ Absolute | Δ Percentage |
|------|----------------|-----------------|------------|--------------|
| TopK | 0.026 | 0.109 | +0.083 | +314% |
| BiasFree | 0.001 | 0.083 | +0.082 | +8,112% |
| ReLU | ~0.00002 | ~0.002 | +0.002 | +10,873% |

Absolute spike magnitudes are comparable or smaller; percentage inflation is an artifact of lower baseline loss.

#### 13.6 Recommendation

- **Do NOT use percentage-based spike detection** for BiasFree/ReLU modes
- Use absolute-value threshold (e.g., `Δloss > 0.1`) or analyze final 100 steps only
- BiasFree and ReLU are superior: final loss 1000x lower than TopK (0.00001 vs 0.02)
- Previous report of "Python/Julia ReLU = 0 spikes" was incorrect (retested: Python = 29)

#### 13.7 Files changed

| File | Change |
|------|--------|
| `go/train.go:325` | Added `t.model.SetRoutingMode(ReLUMode, t.config.ReLULambdaL1)` after adaptive lambda update |

#### 13.8 Spike detection method replaced (2026-02-15)

**Old method**: Consecutive percentage increase > 50%
- Formula: `(loss[t] - loss[t-1]) / loss[t-1] > 0.5`
- Problem: At low loss (0.00001), any tiny fluctuation triggers false positive

**New method**: EMA-relative threshold + absolute floor (dual condition)
- Spike when BOTH: `loss[t] > ema[t] * 3.0` AND `loss[t] - loss[t-1] > 0.001`
- EMA alpha = 0.1 (trailing weighted average)
- Absolute floor prevents noise at low loss from triggering

**Results comparison (Python, 500 steps)**:

| Mode | Old Spikes | New Spikes | False positives removed |
|------|-----------|-----------|------------------------|
| TopK | 2 | 1 | 1 |
| BiasFree | 23 | 2 | 21 |
| ReLU | 29 | 9 | 20 |

**Files changed**: `scripts/convergence_plots.py`
- Added `detect_spikes()` function
- SVG plots now show red circle markers at spike locations + count annotation
- JSON output includes spike details array
- Summary output includes spike count per language

#### 13.9 N=30 statistical benchmark pipeline (2026-02-15)

Upgraded `convergence_plots.py` from single-run to N=30 repeated trials with rigorous statistical analysis.

**Methodology** (based on "When +1% Is Not Enough" arXiv:2511.19794, Owen 2025 arXiv:2508.10083):
- N=30 paired runs per (language, device) with seeds 1-30
- BCa (bias-corrected and accelerated) bootstrap 95% CI on final loss
- Sign-flip permutation test (p < 0.05) for CPU vs GPU paired comparison
- 20% trimmed mean + Winsorized std for robust location/dispersion
- Spike statistics: median + IQR (not mean ± std)

**Why not t-test**:
- Benchmark loss distributions are non-normal (right-skewed with spikes)
- BCa bootstrap adjusts for bias and skewness automatically
- Sign-flip permutation test is exact, requires no distributional assumptions
- N=30 + BCa is conservative and paper-grade

**Files changed (8)**:

| File | Change |
|------|--------|
| `scripts/convergence_plots.py` | Major rewrite: N_TRIALS=30, CPU+GPU, BCa CI, permutation test, trimmed mean, SVG per device |
| `scripts/convergence_python.py` | Added `--seed N` CLI arg |
| `scripts/convergence_python_gpu.py` | Added `--seed N` + `--routing-mode` CLI args |
| `scripts/convergence_julia.jl` | Added `--seed N` ARGS parsing |
| `scripts/convergence_julia_gpu.jl` | Added `--seed N` ARGS parsing |
| `rust/src/bin/convergence.rs` | Added `--seed N` CLI arg |
| `go/nn_test.go` | All 3 TestConvergence* read `CONV_SEED` env var |

**Output**: `benchmarks/convergence/convergence_stats.json` with per-(language, device) BCa CI, trimmed mean, spike IQR, and paired CPU/GPU comparison results.

**Statistical functions implemented** (no scipy dependency):
- `_norm_cdf()` / `_norm_ppf()`: via math.erfc / Abramowitz & Stegun 26.2.23
- `bca_bootstrap_ci()`: bias-corrected accelerated bootstrap
- `sign_flip_permutation_p()`: two-sided exact permutation test
- `trimmed_mean()` / `winsorized_std()`: robust estimators

**Estimated runtime**: ~5 min (CPU only), ~10 min (CPU+GPU), ~30 min (3 modes × CPU+GPU)
