# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""CPU-only benchmark for Python MoE Transformer.

Imports exclusively from python.cpu -- no Metal or PyObjC dependencies.

Measures:
  1. Forward pass (tiny config)
  2. Training step (tiny config)
  3. Kernel micro-benchmarks (matmul, softmax, rmsnorm)
  4. Parallel forward (ProcessPoolExecutor)
  5. Scale comparison (hidden=256/512 forward/train)
"""

import gc
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from python.cpu.config import Config
from python.cpu.tensor import Tensor, seed_rng
from python.cpu.model import MoETransformer
from python.cpu.train import Trainer, TrainConfig


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


# Constants
N_TRIALS = _env_int("ROSETTA_BENCH_TRIALS", 10)
N_WARMUP = _env_int("ROSETTA_BENCH_WARMUP", 3)
SEED = 42
VOCAB = 1000
HIDDEN = 64

# GC tracking
_gc_time_acc = 0
_gc_start_time = 0


def _gc_callback(phase, info):
    global _gc_time_acc, _gc_start_time
    if phase == "start":
        _gc_start_time = time.perf_counter_ns()
    elif phase == "stop":
        _gc_time_acc += time.perf_counter_ns() - _gc_start_time


gc.callbacks.append(_gc_callback)


def _log(msg):
    print(msg, file=sys.stderr)


def _percentile(sorted_vals, p):
    n = len(sorted_vals)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _check_numerical(output):
    if isinstance(output, Tensor):
        arr = output.data
    elif isinstance(output, np.ndarray):
        arr = output
    else:
        return 0, 0, 0.0
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0
    return nan_count, inf_count, max_abs


def _make_input(batch, seq, vocab=VOCAB):
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq, dtype=np.float32)[None, :]
    return Tensor.from_numpy(((b * seq + s) % vocab).astype(np.float32))


def _make_targets(batch, seq, vocab=VOCAB):
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq, dtype=np.float32)[None, :]
    return Tensor.from_numpy(((b * seq + s + 1) % vocab).astype(np.float32))


def _measure(setup_fn, run_fn, scenario_id, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    global _gc_time_acc
    _log(f"  [{scenario_id}] setup...")
    ctx = setup_fn()

    _log(f"  [{scenario_id}] warmup x{n_warmup}")
    warmup_timings_ns = []
    for _ in range(n_warmup):
        t0 = time.perf_counter_ns()
        run_fn(ctx)
        t1 = time.perf_counter_ns()
        warmup_timings_ns.append(t1 - t0)

    gc.collect()

    gc_pause_before = sum(s.get("collections", 0) for s in gc.get_stats())
    _gc_time_acc = 0
    tracemalloc.start()

    timings_ns = []
    cpu_times_ns = []
    last_output = None
    _log(f"  [{scenario_id}] measuring x{n_trials}")
    for _ in range(n_trials):
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter_ns()
        last_output = run_fn(ctx)
        t1 = time.perf_counter_ns()
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        timings_ns.append(t1 - t0)
        cpu_ns = int(
            ((ru_after.ru_utime + ru_after.ru_stime)
             - (ru_before.ru_utime + ru_before.ru_stime)) * 1e9
        )
        cpu_times_ns.append(cpu_ns)

    gc_wall_time_ns = _gc_time_acc
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gc_pause_after = sum(s.get("collections", 0) for s in gc.get_stats())

    snap_after = tracemalloc.take_snapshot()
    stats = snap_after.compare_to(tracemalloc.take_snapshot(), "lineno")
    alloc_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    tracemalloc.stop()

    sorted_t = sorted(timings_ns)
    nan_count, inf_count, max_abs = _check_numerical(last_output)
    gc_collections = gc_pause_after - gc_pause_before
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    median_ns = int(median(timings_ns))
    cpu_median_ns = int(median(cpu_times_ns))

    median_sec = median_ns / 1e9
    alloc_rate = int(alloc_bytes / median_sec) if median_sec > 0 else 0
    sum_timings = sum(timings_ns)
    gc_throughput = 1.0 - (gc_wall_time_ns / sum_timings) if sum_timings > 0 else 1.0

    return {
        "timings_ns": timings_ns,
        "cpu_times_ns": cpu_times_ns,
        "warmup_timings_ns": warmup_timings_ns,
        "median_ns": median_ns,
        "p95_ns": int(_percentile(sorted_t, 95)),
        "min_ns": min(timings_ns),
        "max_ns": max(timings_ns),
        "iqr_ns": q3 - q1,
        "cpu_median_ns": cpu_median_ns,
        "memory": {"peak_rss_bytes": rss, "alloc_bytes": alloc_bytes},
        "gc": {"total_gc_time_ns": gc_wall_time_ns, "gc_pause_count": gc_collections},
        "numerical": {"nan_count": nan_count, "inf_count": inf_count, "max_abs": max_abs},
        "derived": {"alloc_rate_bytes_per_sec": alloc_rate, "gc_throughput": gc_throughput},
    }


def _add_throughput(result, batch, seq_len):
    median_sec = result["median_ns"] / 1e9
    if median_sec > 0:
        result["throughput_tokens_per_sec"] = (batch * seq_len) / median_sec


# --- Scenarios ---

def scenario_forward():
    """CPU forward pass (tiny config)."""
    batch, seq = 2, 32

    def setup():
        seed_rng(SEED)
        return MoETransformer(Config.tiny()), _make_input(batch, seq)

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, "cpu_forward")
    result.update(id="cpu_forward", axis="cpu", params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def scenario_train_step():
    """CPU training step."""
    batch, seq = 2, 8

    def setup():
        seed_rng(SEED)
        model = MoETransformer(Config.tiny())
        trainer = Trainer(model, TrainConfig.default())
        return trainer, _make_input(batch, seq), _make_targets(batch, seq)

    def run(ctx):
        trainer, x, t = ctx
        return trainer.train_step(x, t)

    result = _measure(setup, run, "cpu_train_step")
    result.update(id="cpu_train_step", axis="cpu",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def scenario_kernel_matmul():
    """CPU matmul kernel (BLAS)."""
    M = K = N = 64
    known_flops = 2 * M * N * K

    def setup():
        np.random.seed(SEED)
        a = np.random.randn(M, K).astype(np.float32)
        b_t = np.random.randn(N, K).astype(np.float32).T.copy()
        out = np.empty((M, N), dtype=np.float32)
        return a, b_t, out

    def run(ctx):
        a, b_t, out = ctx
        np.matmul(a, b_t, out=out)
        return out

    result = _measure(setup, run, "cpu_kernel_matmul")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(id="cpu_kernel_matmul", axis="cpu_kernel",
                  params={"M": M, "K": K, "N": N}, known_flops=known_flops,
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


def scenario_kernel_softmax():
    """CPU softmax kernel."""
    n = 1000
    known_flops = 4 * n

    def setup():
        np.random.seed(SEED)
        x = np.random.randn(n).astype(np.float32)
        buf = np.empty_like(x)
        return x, buf

    def run(ctx):
        x, buf = ctx
        np.copyto(buf, x)
        buf -= np.max(buf)
        np.exp(buf, out=buf)
        buf /= np.sum(buf)
        return buf

    result = _measure(setup, run, "cpu_kernel_softmax")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(id="cpu_kernel_softmax", axis="cpu_kernel",
                  params={"n": n}, known_flops=known_flops,
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


def scenario_kernel_rmsnorm():
    """CPU rmsnorm kernel."""
    shape = (2, 32, 64)
    known_flops = 2 * 32 * 64 * 3
    eps = 1e-6

    def setup():
        np.random.seed(SEED)
        x = np.random.randn(*shape).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        return x, weight

    def run(ctx):
        x, weight = ctx
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    result = _measure(setup, run, "cpu_kernel_rmsnorm")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(id="cpu_kernel_rmsnorm", axis="cpu_kernel",
                  params={"shape": list(shape)}, known_flops=known_flops,
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


# --- Parallel ---
_worker_state = {}


def _parallel_init_worker(batch, seq_len, vocab, seed):
    worker_seed = seed + os.getpid() % 1000
    np.random.seed(worker_seed)
    seed_rng(worker_seed)
    model = MoETransformer(Config.tiny())
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq_len, dtype=np.float32)[None, :]
    x = Tensor.from_numpy(((b * seq_len + s) % vocab).astype(np.float32))
    _worker_state["model"] = model
    _worker_state["input"] = x


def _parallel_forward_worker(_unused):
    _worker_state["model"].forward(_worker_state["input"])


def scenario_parallel(n_procs):
    batch, seq = 2, 32
    sid = f"cpu_parallel_T{n_procs}"
    timings_ns = []
    warmup_timings_ns = []
    cpu_times_ns = []

    _log(f"  [{sid}] creating pool (n_procs={n_procs})...")
    pool = ProcessPoolExecutor(
        max_workers=n_procs,
        initializer=_parallel_init_worker,
        initargs=(batch, seq, VOCAB, SEED),
    )
    list(pool.map(_parallel_forward_worker, range(n_procs)))

    _log(f"  [{sid}] warmup x{N_WARMUP}")
    for _ in range(N_WARMUP):
        t0 = time.perf_counter_ns()
        list(pool.map(_parallel_forward_worker, range(n_procs)))
        t1 = time.perf_counter_ns()
        warmup_timings_ns.append(t1 - t0)

    _log(f"  [{sid}] measuring x{N_TRIALS}")
    for _ in range(N_TRIALS):
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter_ns()
        list(pool.map(_parallel_forward_worker, range(n_procs)))
        t1 = time.perf_counter_ns()
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        timings_ns.append(t1 - t0)
        cpu_ns = int(
            ((ru_after.ru_utime + ru_after.ru_stime)
             - (ru_before.ru_utime + ru_before.ru_stime)) * 1e9
        )
        cpu_times_ns.append(cpu_ns)

    pool.shutdown(wait=True)

    sorted_t = sorted(timings_ns)
    median_ns = int(median(timings_ns))
    median_sec = median_ns / 1e9
    throughput = (n_procs * batch * seq) / median_sec if median_sec > 0 else 0.0
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "id": sid,
        "axis": "cpu_parallel",
        "params": {"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN, "thread_count": n_procs},
        "parallel_semantics": "independent",
        "timings_ns": timings_ns,
        "cpu_times_ns": cpu_times_ns,
        "warmup_timings_ns": warmup_timings_ns,
        "median_ns": median_ns,
        "p95_ns": int(_percentile(sorted_t, 95)),
        "min_ns": min(timings_ns),
        "max_ns": max(timings_ns),
        "iqr_ns": q3 - q1,
        "cpu_median_ns": int(median(cpu_times_ns)),
        "throughput_tokens_per_sec": throughput,
        "memory": {"peak_rss_bytes": rss, "alloc_bytes": None},
        "gc": {"total_gc_time_ns": None, "gc_pause_count": None},
        "numerical": {"nan_count": 0, "inf_count": 0, "max_abs": 0.0},
        "derived": {"alloc_rate_bytes_per_sec": None, "gc_throughput": None},
        "warmup_runs": N_WARMUP,
        "trial_runs": N_TRIALS,
    }


def scenario_scale_forward_256():
    """CPU forward pass (hidden=256)."""
    batch, seq = 2, 32

    def setup():
        seed_rng(SEED)
        return MoETransformer(Config.small()), _make_input(batch, seq)

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, "scale_forward_256")
    result.update(id="scale_forward_256", axis="scale",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": 256},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def scenario_scale_train_256():
    """CPU training step (hidden=256)."""
    batch, seq = 2, 8

    def setup():
        seed_rng(SEED)
        model = MoETransformer(Config.small())
        trainer = Trainer(model, TrainConfig.default())
        return trainer, _make_input(batch, seq), _make_targets(batch, seq)

    def run(ctx):
        trainer, x, t = ctx
        return trainer.train_step(x, t)

    result = _measure(setup, run, "scale_train_256")
    result.update(id="scale_train_256", axis="scale",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": 256},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def scenario_scale_forward_512():
    """CPU forward pass (hidden=512)."""
    batch, seq = 2, 32

    def setup():
        seed_rng(SEED)
        return MoETransformer(Config.medium()), _make_input(batch, seq)

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, "scale_forward_512")
    result.update(id="scale_forward_512", axis="scale",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": 512},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def scenario_scale_train_512():
    """CPU training step (hidden=512)."""
    batch, seq = 2, 8

    def setup():
        seed_rng(SEED)
        model = MoETransformer(Config.medium())
        trainer = Trainer(model, TrainConfig.default())
        return trainer, _make_input(batch, seq), _make_targets(batch, seq)

    def run(ctx):
        trainer, x, t = ctx
        return trainer.train_step(x, t)

    result = _measure(setup, run, "scale_train_512")
    result.update(id="scale_train_512", axis="scale",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": 512},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def _metadata():
    return {
        "language": "python",
        "backend": "cpu",
        "language_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "cpu_model": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_warmup": N_WARMUP,
    }


def main():
    _log("Python MoE Benchmark -- CPU only (from python.cpu)")
    np.random.seed(SEED)

    scenarios = []
    total = 13

    _log(f"[1/{total}] cpu_forward")
    scenarios.append(scenario_forward())

    _log(f"[2/{total}] cpu_train_step")
    scenarios.append(scenario_train_step())

    _log(f"[3/{total}] cpu_kernel_matmul")
    scenarios.append(scenario_kernel_matmul())

    _log(f"[4/{total}] cpu_kernel_softmax")
    scenarios.append(scenario_kernel_softmax())

    _log(f"[5/{total}] cpu_kernel_rmsnorm")
    scenarios.append(scenario_kernel_rmsnorm())

    for i, t in enumerate([1, 2, 4], start=6):
        _log(f"[{i}/{total}] cpu_parallel_T{t}")
        # Skip parallel in quick mode if env says so
        if os.environ.get("ROSETTA_QUICK") == "1" and t > 1:
            continue
        scenarios.append(scenario_parallel(t))

    _log("[9/13] scale_forward_256")
    scenarios.append(scenario_scale_forward_256())
    _log("[10/13] scale_train_256")
    scenarios.append(scenario_scale_train_256())
    _log("[11/13] scale_forward_512")
    scenarios.append(scenario_scale_forward_512())
    _log("[12/13] scale_train_512")
    scenarios.append(scenario_scale_train_512())

    _log(f"[{total}/{total}] done")

    result = {"metadata": _metadata(), "scenarios": scenarios}
    _log("Writing JSON to stdout.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
