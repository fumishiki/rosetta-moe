# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""CPU benchmark harness for Python MoE Transformer -- 5-axis design (22 scenarios).

Measures performance across five axes:
  1. Memory Management   -- training step, batch/seq scaling (11 scenarios)
  2. Compiler Optimization -- kernel micro-benchmarks: matmul, softmax, rmsnorm (3)
  3. Type System         -- warm vs cold dispatch overhead (2)
  4. Parallel            -- ProcessPoolExecutor scaling (1/2/4 workers) forward + train (6)
  5. Scale Comparison    -- hidden=256 forward/train (2)

Measurement methodology:
  - Wall time: time.perf_counter_ns (monotonic, ~ns resolution)
  - CPU time:  resource.getrusage (user + system)
  - Memory:    tracemalloc snapshots (Python heap only -- see caveat below)
  - GC:        gc.callbacks for wall-time tracking + gc.get_stats for collection counts
  - Numerical: NaN/Inf/max_abs checks on outputs

IMPORTANT caveat on tracemalloc:
  tracemalloc only tracks allocations made through Python's memory allocator
  (PyMem_Malloc / PyObject_Malloc).  NumPy's large array data is allocated via
  libc malloc (or the system allocator), which tracemalloc does NOT see.
  Therefore alloc_bytes significantly undercounts actual memory usage.
  Use peak_rss_bytes (from getrusage) for a more accurate picture.

Parallelism note:
  Python's GIL prevents true thread parallelism for CPU-bound Python code.
  We use ProcessPoolExecutor to fork separate processes, each with its own
  GIL.  The pool is pre-created with models initialized in each worker
  (via initializer=), so trials only measure dispatch + forward pass,
  matching the methodology used in Go/Rust/Julia benchmarks.
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

from python.config import Config
from python.tensor import Tensor
from python.model import MoETransformer
from python.train import Trainer, TrainConfig

# Constants
N_TRIALS = 10
N_WARMUP = 3
SEED = 42
VOCAB = 1000
HIDDEN = 64

# ---------------------------------------------------------------------------
# GC wall-time measurement via gc.callbacks
# ---------------------------------------------------------------------------
# gc.callbacks fires "start"/"stop" around every collection.  We accumulate
# the wall-clock time spent inside GC to compute gc_throughput later.
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
    """Linear interpolation percentile on a pre-sorted list."""
    n = len(sorted_vals)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _check_numerical(output):
    """Check for NaN/Inf in output tensor or array."""
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
    """Create deterministic input tensor from batch/seq indices."""
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq, dtype=np.float32)[None, :]
    return Tensor.from_numpy(((b * seq + s) % vocab).astype(np.float32))


def _make_targets(batch, seq, vocab=VOCAB):
    """Create deterministic target tensor (shifted by 1 from input)."""
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq, dtype=np.float32)[None, :]
    return Tensor.from_numpy(((b * seq + s + 1) % vocab).astype(np.float32))


def _measure(setup_fn, run_fn, scenario_id, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    """Core measurement harness.

    1. Runs setup_fn() once to create the context (model, inputs, etc.)
    2. Warmup: runs run_fn(ctx) n_warmup times (not measured)
    3. Measurement: runs run_fn(ctx) n_trials times with timing/memory/GC tracking
    4. Returns a dict of statistics

    Memory tracking uses tracemalloc (Python heap only).  NumPy array data
    buffers are NOT tracked -- see module docstring for details.
    """
    global _gc_time_acc
    _log(f"  [{scenario_id}] setup...")
    ctx = setup_fn()

    # Warmup (stabilizes JIT-like effects in numpy, populates caches)
    _log(f"  [{scenario_id}] warmup x{n_warmup}")
    warmup_timings_ns = []
    for _ in range(n_warmup):
        t0 = time.perf_counter_ns()
        run_fn(ctx)
        t1 = time.perf_counter_ns()
        warmup_timings_ns.append(t1 - t0)

    gc.collect()

    # Pre-measurement state
    gc_pause_before = sum(s.get("collections", 0) for s in gc.get_stats())
    _gc_time_acc = 0
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

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
            (
                (ru_after.ru_utime + ru_after.ru_stime)
                - (ru_before.ru_utime + ru_before.ru_stime)
            )
            * 1e9
        )
        cpu_times_ns.append(cpu_ns)

    gc_wall_time_ns = _gc_time_acc

    snap_after = tracemalloc.take_snapshot()
    # peak_rss_bytes from getrusage is the high-water RSS mark for the process.
    # On macOS this is in bytes; on Linux it's in kilobytes (platform-dependent).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gc_pause_after = sum(s.get("collections", 0) for s in gc.get_stats())

    # tracemalloc diff only captures Python-heap allocations (PyMem/PyObject).
    # NumPy array data (allocated via libc malloc) is invisible here.
    stats = snap_after.compare_to(snap_before, "lineno")
    alloc_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
    tracemalloc.stop()

    sorted_t = sorted(timings_ns)
    sorted_cpu = sorted(cpu_times_ns)
    nan_count, inf_count, max_abs = _check_numerical(last_output)
    gc_collections = gc_pause_after - gc_pause_before
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    median_ns = int(median(timings_ns))
    cpu_median_ns = int(median(cpu_times_ns))

    # Derived metrics
    median_sec = median_ns / 1e9
    alloc_rate = int(alloc_bytes / median_sec) if median_sec > 0 else 0
    sum_timings = sum(timings_ns)
    # gc_throughput: fraction of wall time NOT spent in GC (1.0 = no GC overhead)
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
        "memory": {
            "peak_rss_bytes": rss,
            "alloc_bytes": alloc_bytes,
        },
        "gc": {
            "total_gc_time_ns": gc_wall_time_ns,
            "gc_pause_count": gc_collections,
        },
        "numerical": {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "max_abs": max_abs,
        },
        "derived": {
            "alloc_rate_bytes_per_sec": alloc_rate,
            "gc_throughput": gc_throughput,
        },
    }


def _add_throughput(result, batch, seq_len):
    """Add tokens/sec throughput metric to result dict."""
    median_sec = result["median_ns"] / 1e9
    if median_sec > 0:
        result["throughput_tokens_per_sec"] = (batch * seq_len) / median_sec


# ---------------------------------------------------------------------------
# Axis 1: Memory Management
# ---------------------------------------------------------------------------

def scenario_mem_train_step():
    """Benchmark a single training step (forward + backward + AdamW)."""
    batch, seq = 2, 8

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.tiny())
        trainer = Trainer(model, TrainConfig.default())
        input_ids = _make_input(batch, seq)
        targets = _make_targets(batch, seq)
        return trainer, input_ids, targets

    def run(ctx):
        trainer, input_ids, targets = ctx
        return trainer.train_step(input_ids, targets)

    result = _measure(setup, run, "mem_train_step")
    result.update(
        id="mem_train_step",
        axis="memory",
        params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


def _scenario_mem_scale_batch(batch):
    """Benchmark forward pass scaling with batch size."""
    seq = 32
    sid = f"mem_scale_batch_{batch}"

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.tiny())
        x = _make_input(batch, seq)
        return model, x

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, sid)
    result.update(
        id=sid,
        axis="memory",
        params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


def _scenario_mem_scale_seq(seq):
    """Benchmark forward pass scaling with sequence length."""
    batch = 2
    sid = f"mem_scale_seq_{seq}"

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.tiny())
        x = _make_input(batch, seq)
        return model, x

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, sid)
    result.update(
        id=sid,
        axis="memory",
        params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


# ---------------------------------------------------------------------------
# Axis 2: Compiler Optimization (kernel micro-benchmarks)
# ---------------------------------------------------------------------------

def scenario_kernel_matmul():
    """Benchmark raw matmul (np.matmul -> BLAS sgemm via Accelerate)."""
    M = K = N = 64
    known_flops = 2 * M * N * K  # 524288

    def setup():
        np.random.seed(SEED)
        a = np.random.randn(M, K).astype(np.float32)
        # Pre-transpose b so run() only times np.matmul (BLAS), not transposition
        b_t = np.random.randn(N, K).astype(np.float32).T.copy()
        # Pre-allocate output to avoid allocation in timed region
        out = np.empty((M, N), dtype=np.float32)
        return a, b_t, out

    def run(ctx):
        a, b_t, out = ctx
        # Pure BLAS matmul -- no Tensor wrapper, no transpose
        np.matmul(a, b_t, out=out)
        return out

    result = _measure(setup, run, "kernel_matmul")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(
        id="kernel_matmul",
        axis="compiler",
        params={"M": M, "K": K, "N": N},
        known_flops=known_flops,
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    return result


def scenario_kernel_softmax():
    """Benchmark pure-NumPy softmax (shift-exp-normalize)."""
    n = 1000
    # Approximate FLOPs: max(n) + sub(n) + exp(n) + sum(n) + div(n) ~ 4n
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

    result = _measure(setup, run, "kernel_softmax")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(
        id="kernel_softmax",
        axis="compiler",
        params={"n": n},
        known_flops=known_flops,
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    return result


def scenario_kernel_rmsnorm():
    """Benchmark pure-NumPy RMSNorm kernel."""
    shape = (2, 32, 64)
    # Approximate FLOPs: sq(n) + mean(n) + sqrt(1) + div(n) + mul(n) ~ 3n per element
    known_flops = 2 * 32 * 64 * 3  # 12288
    eps = 1e-6

    def setup():
        np.random.seed(SEED)
        x = np.random.randn(*shape).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        return x, weight

    def run(ctx):
        x, weight = ctx
        # RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    result = _measure(setup, run, "kernel_rmsnorm")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(
        id="kernel_rmsnorm",
        axis="compiler",
        params={"shape": list(shape)},
        known_flops=known_flops,
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    return result


# ---------------------------------------------------------------------------
# Axis 3: Type System (dispatch)
# ---------------------------------------------------------------------------

def scenario_dispatch_warm():
    """Benchmark forward pass with pre-created model (warm dispatch)."""
    batch, seq = 2, 32

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.tiny())
        x = _make_input(batch, seq)
        return model, x

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, "dispatch_warm")
    result.update(
        id="dispatch_warm",
        axis="type_system",
        params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


def scenario_dispatch_cold():
    """Benchmark forward pass with model creation inside timed region (cold dispatch).

    Measures the overhead of Python object construction + __init__ chains,
    which is the Python equivalent of "cold start" / first-call overhead
    in compiled languages.
    """
    batch, seq = 1, 8

    def setup():
        np.random.seed(SEED)
        x = _make_input(batch, seq)
        return (x,)

    def run(ctx):
        (x,) = ctx
        model = MoETransformer(Config.tiny())
        return model.forward(x)

    result = _measure(setup, run, "dispatch_cold", n_warmup=N_WARMUP)
    result.update(
        id="dispatch_cold",
        axis="type_system",
        params={"batch": batch, "seq_len": seq, "hidden_dim": HIDDEN},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


def scenario_scale_forward_256():
    """Benchmark forward pass with hidden=256 (scale comparison)."""
    batch, seq = 2, 32

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.small())
        x = _make_input(batch, seq)
        return model, x

    def run(ctx):
        model, x = ctx
        return model.forward(x)

    result = _measure(setup, run, "scale_forward_256")
    result.update(
        id="scale_forward_256",
        axis="scale",
        params={"batch": batch, "seq_len": seq, "hidden_dim": 256},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


def scenario_scale_train_256():
    """Benchmark training step with hidden=256 (scale comparison)."""
    batch, seq = 2, 8

    def setup():
        np.random.seed(SEED)
        model = MoETransformer(Config.small())
        trainer = Trainer(model, TrainConfig.default())
        input_ids = _make_input(batch, seq)
        targets = _make_targets(batch, seq)
        return trainer, input_ids, targets

    def run(ctx):
        trainer, input_ids, targets = ctx
        return trainer.train_step(input_ids, targets)

    result = _measure(setup, run, "scale_train_256")
    result.update(
        id="scale_train_256",
        axis="scale",
        params={"batch": batch, "seq_len": seq, "hidden_dim": 256},
        warmup_runs=N_WARMUP,
        trial_runs=N_TRIALS,
    )
    _add_throughput(result, batch, seq)
    return result


# ---------------------------------------------------------------------------
# Axis 4: Parallel
# ---------------------------------------------------------------------------

_worker_state = {}


def _parallel_init_worker(batch, seq_len, vocab, seed):
    """Initializer for ProcessPoolExecutor workers.

    Creates model and input tensor once per worker process and stores them
    in module-level _worker_state.  This avoids measuring model construction
    and import overhead in every trial, matching Go/Rust/Julia methodology.
    """
    worker_seed = seed + os.getpid() % 1000
    np.random.seed(worker_seed)
    model = MoETransformer(Config.tiny())
    b = np.arange(batch, dtype=np.float32)[:, None]
    s = np.arange(seq_len, dtype=np.float32)[None, :]
    x = Tensor.from_numpy(((b * seq_len + s) % vocab).astype(np.float32))
    _worker_state["model"] = model
    _worker_state["input"] = x


def _parallel_forward_worker(_unused):
    """Run a single forward pass using the pre-initialized worker state."""
    _worker_state["model"].forward(_worker_state["input"])


def _scenario_parallel(n_procs):
    """Benchmark parallel forward passes using ProcessPoolExecutor.

    The pool is created once with models pre-initialized in each worker
    via the initializer callback.  Warmup and trial loops only measure
    the dispatch + forward pass time, not process creation overhead.
    """
    batch, seq = 2, 32
    sid = f"parallel_T{n_procs}"
    timings_ns = []
    warmup_timings_ns = []
    cpu_times_ns = []

    _log(f"  [{sid}] creating pool (n_procs={n_procs})...")
    pool = ProcessPoolExecutor(
        max_workers=n_procs,
        initializer=_parallel_init_worker,
        initargs=(batch, seq, VOCAB, SEED),
    )
    # Force all workers to initialize by running one forward pass each
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
            (
                (ru_after.ru_utime + ru_after.ru_stime)
                - (ru_before.ru_utime + ru_before.ru_stime)
            )
            * 1e9
        )
        cpu_times_ns.append(cpu_ns)

    pool.shutdown(wait=True)

    sorted_t = sorted(timings_ns)
    sorted_cpu = sorted(cpu_times_ns)
    median_ns = int(median(timings_ns))
    cpu_median_ns = int(median(cpu_times_ns))
    median_sec = median_ns / 1e9
    throughput = (n_procs * batch * seq) / median_sec if median_sec > 0 else 0.0
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "id": sid,
        "axis": "parallel",
        "params": {
            "batch": batch,
            "seq_len": seq,
            "hidden_dim": HIDDEN,
            "thread_count": n_procs,
        },
        "parallel_semantics": "independent",
        "timings_ns": timings_ns,
        "cpu_times_ns": cpu_times_ns,
        "warmup_timings_ns": warmup_timings_ns,
        "median_ns": median_ns,
        "p95_ns": int(_percentile(sorted_t, 95)),
        "min_ns": min(timings_ns),
        "max_ns": max(timings_ns),
        "iqr_ns": q3 - q1,
        "cpu_median_ns": cpu_median_ns,
        "throughput_tokens_per_sec": throughput,
        "memory": {"peak_rss_bytes": rss, "alloc_bytes": None},
        "gc": {"total_gc_time_ns": None, "gc_pause_count": None},
        "numerical": {"nan_count": 0, "inf_count": 0, "max_abs": 0.0},
        "derived": {
            "alloc_rate_bytes_per_sec": None,
            "gc_throughput": None,
        },
        "warmup_runs": N_WARMUP,
        "trial_runs": N_TRIALS,
    }


def _parallel_train_init_worker(batch, seq_len, vocab, seed):
    """Initializer for parallel training workers."""
    worker_seed = seed + os.getpid() % 1000
    np.random.seed(worker_seed)
    model = MoETransformer(Config.tiny())
    trainer = Trainer(model, TrainConfig.default())
    input_ids = _make_input(batch, seq_len, vocab)
    targets = _make_targets(batch, seq_len, vocab)
    _worker_state["trainer"] = trainer
    _worker_state["input"] = input_ids
    _worker_state["targets"] = targets


def _parallel_train_worker(_unused):
    """Run a single training step using pre-initialized worker state."""
    _worker_state["trainer"].train_step(_worker_state["input"], _worker_state["targets"])


def _scenario_parallel_train(n_procs):
    """Benchmark parallel training steps using ProcessPoolExecutor.

    The pool is created once with models and trainers pre-initialized in each
    worker via the initializer callback.  Warmup and trial loops only measure
    the dispatch + train_step time, not process creation overhead.
    """
    batch, seq = 2, 8
    sid = f"parallel_train_T{n_procs}"
    timings_ns = []
    warmup_timings_ns = []
    cpu_times_ns = []

    _log(f"  [{sid}] creating pool (n_procs={n_procs})...")
    pool = ProcessPoolExecutor(
        max_workers=n_procs,
        initializer=_parallel_train_init_worker,
        initargs=(batch, seq, VOCAB, SEED),
    )
    # Force all workers to initialize by running one train step each
    list(pool.map(_parallel_train_worker, range(n_procs)))

    _log(f"  [{sid}] warmup x{N_WARMUP}")
    for _ in range(N_WARMUP):
        t0 = time.perf_counter_ns()
        list(pool.map(_parallel_train_worker, range(n_procs)))
        t1 = time.perf_counter_ns()
        warmup_timings_ns.append(t1 - t0)

    _log(f"  [{sid}] measuring x{N_TRIALS}")
    for _ in range(N_TRIALS):
        ru_before = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter_ns()
        list(pool.map(_parallel_train_worker, range(n_procs)))
        t1 = time.perf_counter_ns()
        ru_after = resource.getrusage(resource.RUSAGE_SELF)
        timings_ns.append(t1 - t0)
        cpu_ns = int(
            (
                (ru_after.ru_utime + ru_after.ru_stime)
                - (ru_before.ru_utime + ru_before.ru_stime)
            )
            * 1e9
        )
        cpu_times_ns.append(cpu_ns)

    pool.shutdown(wait=True)

    sorted_t = sorted(timings_ns)
    sorted_cpu = sorted(cpu_times_ns)
    median_ns = int(median(timings_ns))
    cpu_median_ns = int(median(cpu_times_ns))
    median_sec = median_ns / 1e9
    throughput = (n_procs * batch * seq) / median_sec if median_sec > 0 else 0.0
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return {
        "id": sid,
        "axis": "parallel",
        "params": {
            "batch": batch,
            "seq_len": seq,
            "hidden_dim": HIDDEN,
            "thread_count": n_procs,
            "workload": "train_step",
        },
        "parallel_semantics": "independent",
        "timings_ns": timings_ns,
        "cpu_times_ns": cpu_times_ns,
        "warmup_timings_ns": warmup_timings_ns,
        "median_ns": median_ns,
        "p95_ns": int(_percentile(sorted_t, 95)),
        "min_ns": min(timings_ns),
        "max_ns": max(timings_ns),
        "iqr_ns": q3 - q1,
        "cpu_median_ns": cpu_median_ns,
        "throughput_tokens_per_sec": throughput,
        "memory": {"peak_rss_bytes": rss, "alloc_bytes": None},
        "gc": {"total_gc_time_ns": None, "gc_pause_count": None},
        "numerical": {"nan_count": 0, "inf_count": 0, "max_abs": 0.0},
        "derived": {
            "alloc_rate_bytes_per_sec": None,
            "gc_throughput": None,
        },
        "warmup_runs": N_WARMUP,
        "trial_runs": N_TRIALS,
    }


# ---------------------------------------------------------------------------
# Metadata & main
# ---------------------------------------------------------------------------

def _metadata():
    """Collect system metadata for the benchmark report."""
    return {
        "language": "python",
        "language_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "cpu_model": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_warmup": N_WARMUP,
    }


def main():
    _log("Python MoE Benchmark -- 5-axis (22 scenarios)")
    np.random.seed(SEED)

    scenarios = []
    total = 22
    idx = 0

    # Axis 1: Memory Management (11 scenarios)
    idx += 1
    _log(f"[{idx}/{total}] mem_train_step")
    scenarios.append(scenario_mem_train_step())

    for bs in [1, 2, 4, 8]:
        idx += 1
        _log(f"[{idx}/{total}] mem_scale_batch_{bs}")
        scenarios.append(_scenario_mem_scale_batch(bs))

    for sl in [8, 16, 32, 64]:
        idx += 1
        _log(f"[{idx}/{total}] mem_scale_seq_{sl}")
        scenarios.append(_scenario_mem_scale_seq(sl))

    # Axis 2: Compiler Optimization (3 scenarios)
    idx += 1
    _log(f"[{idx}/{total}] kernel_matmul")
    scenarios.append(scenario_kernel_matmul())

    idx += 1
    _log(f"[{idx}/{total}] kernel_softmax")
    scenarios.append(scenario_kernel_softmax())

    idx += 1
    _log(f"[{idx}/{total}] kernel_rmsnorm")
    scenarios.append(scenario_kernel_rmsnorm())

    # Axis 3: Type System (2 scenarios)
    idx += 1
    _log(f"[{idx}/{total}] dispatch_warm")
    scenarios.append(scenario_dispatch_warm())

    idx += 1
    _log(f"[{idx}/{total}] dispatch_cold")
    scenarios.append(scenario_dispatch_cold())

    # Axis 4: Parallel (3 scenarios)
    for t in [1, 2, 4]:
        idx += 1
        _log(f"[{idx}/{total}] parallel_T{t}")
        scenarios.append(_scenario_parallel(t))

    # Axis 4b: Parallel Training (3 scenarios)
    for t in [1, 2, 4]:
        idx += 1
        _log(f"[{idx}/{total}] parallel_train_T{t}")
        scenarios.append(_scenario_parallel_train(t))

    # Axis 5: Scale Comparison (2 scenarios)
    idx += 1
    _log(f"[{idx}/{total}] scale_forward_256")
    scenarios.append(scenario_scale_forward_256())

    idx += 1
    _log(f"[{idx}/{total}] scale_train_256")
    scenarios.append(scenario_scale_train_256())

    result = {
        "metadata": _metadata(),
        "scenarios": scenarios,
    }

    _log("Done. Writing JSON to stdout.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
