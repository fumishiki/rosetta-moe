# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""GPU-only (Metal) benchmark for Python MoE Transformer.

Imports from python.gpu -- requires PyObjC Metal/MPS on macOS.

Measures:
  1. GPU kernel micro-benchmarks (MPS matmul, MSL softmax, MSL rmsnorm)
  2. Full-model GPU forward at 3 scales
  3. Full-model GPU train proxy (forward + CE loss sum, no readback) at 3 scales

Policy:
  - Setup uploads model/input/target buffers once.
  - Timed loops do not permit CPU<->GPU transfer APIs.
"""

import gc
import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _log(msg):
    print(msg, file=sys.stderr)


# Check Metal availability before proceeding
try:
    from python.gpu import (
        metal_available,
        MetalContext,
        MetalTensor,
        MetalMoETransformer,
        MetalTrainer,
        seed_rng as gpu_seed_rng,
    )
    from python.config import Config

    if not metal_available():
        _log("Metal not available on this system. Exiting.")
        sys.exit(0)
except ImportError as e:
    _log(f"GPU module import failed: {e}")
    _log("Install: pip install pyobjc-framework-Metal pyobjc-framework-MetalPerformanceShaders")
    sys.exit(1)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


_ORIG_TO_NUMPY = MetalTensor.to_numpy
_ORIG_FROM_NUMPY = MetalTensor.__dict__["from_numpy"]
_ORIG_NUMPY_VIEW = MetalTensor.numpy_view


def _install_transfer_guard() -> None:
    """Ban host transfer APIs during warmup/timed loops."""
    def _blocked_to_numpy(*_args, **_kwargs):
        raise RuntimeError("CPU<->GPU transfer is forbidden in bench_gpu.py timed loop (to_numpy)")

    def _blocked_from_numpy(*_args, **_kwargs):
        raise RuntimeError("CPU<->GPU transfer is forbidden in bench_gpu.py timed loop (from_numpy)")

    def _blocked_numpy_view(*_args, **_kwargs):
        raise RuntimeError("CPU<->GPU transfer is forbidden in bench_gpu.py timed loop (numpy_view)")

    MetalTensor.to_numpy = _blocked_to_numpy
    MetalTensor.from_numpy = classmethod(_blocked_from_numpy)
    MetalTensor.numpy_view = _blocked_numpy_view


def _remove_transfer_guard() -> None:
    MetalTensor.to_numpy = _ORIG_TO_NUMPY
    MetalTensor.from_numpy = _ORIG_FROM_NUMPY
    MetalTensor.numpy_view = _ORIG_NUMPY_VIEW


# Constants
N_TRIALS = _env_int("ROSETTA_BENCH_TRIALS", 10)
N_WARMUP = _env_int("ROSETTA_BENCH_WARMUP", 3)
SEED = 42
VOCAB = 1000

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


def _percentile(sorted_vals, p):
    n = len(sorted_vals)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _measure(setup_fn, run_fn, scenario_id, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    global _gc_time_acc
    _log(f"  [{scenario_id}] setup...")
    ctx = setup_fn()
    _install_transfer_guard()
    try:
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
    finally:
        _remove_transfer_guard()

    gc_wall_time_ns = _gc_time_acc
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gc_pause_after = sum(s.get("collections", 0) for s in gc.get_stats())
    tracemalloc.stop()

    sorted_t = sorted(timings_ns)
    gc_collections = gc_pause_after - gc_pause_before
    q1 = int(_percentile(sorted_t, 25))
    q3 = int(_percentile(sorted_t, 75))
    median_ns = int(median(timings_ns))
    cpu_median_ns = int(median(cpu_times_ns))
    median_sec = median_ns / 1e9
    sum_timings = sum(timings_ns)
    gc_throughput = 1.0 - (gc_wall_time_ns / sum_timings) if sum_timings > 0 else 1.0

    # Check numerical output
    nan_count = inf_count = 0
    max_abs = 0.0
    if isinstance(last_output, np.ndarray):
        nan_count = int(np.isnan(last_output).sum())
        inf_count = int(np.isinf(last_output).sum())
        finite = last_output[np.isfinite(last_output)]
        max_abs = float(np.max(np.abs(finite))) if finite.size > 0 else 0.0

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
        "memory": {"peak_rss_bytes": rss},
        "gc": {"total_gc_time_ns": gc_wall_time_ns, "gc_pause_count": gc_collections},
        "numerical": {"nan_count": nan_count, "inf_count": inf_count, "max_abs": max_abs},
        "derived": {"gc_throughput": gc_throughput},
    }


def _add_throughput(result, batch, seq_len):
    median_sec = result["median_ns"] / 1e9
    if median_sec > 0:
        result["throughput_tokens_per_sec"] = (batch * seq_len) / median_sec


# --- GPU Kernel Benchmarks ---

def scenario_gpu_kernel_matmul():
    """GPU MPS matmul 256x256."""
    m, n, k = 256, 256, 256
    known_flops = 2 * m * n * k

    def setup():
        ctx = MetalContext()
        a = MetalTensor.empty(ctx, [m, k])
        b = MetalTensor.empty(ctx, [k, n])
        c = MetalTensor.empty(ctx, [m, n])
        return ctx, a, b, c

    def run(ctx_tuple):
        ctx, a, b, c = ctx_tuple
        ctx.mps_matmul(a, b, c, m, n, k)
        return None

    result = _measure(setup, run, "gpu_kernel_matmul")
    median_sec = result["median_ns"] / 1e9
    gflops = (known_flops / median_sec / 1e9) if median_sec > 0 else 0.0
    result["derived"]["gflops"] = gflops
    result.update(id="gpu_kernel_matmul", axis="gpu",
                  params={"m": m, "n": n, "k": k}, known_flops=known_flops,
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


def scenario_gpu_kernel_softmax():
    """GPU MSL softmax kernel."""
    n = 1000

    def setup():
        ctx = MetalContext()
        inp = MetalTensor.empty(ctx, [1, n])
        out = MetalTensor.empty(ctx, [1, n])
        return ctx, inp, out

    def run(ctx_tuple):
        ctx, inp, out = ctx_tuple
        ctx.dispatch_kernel("softmax", [inp, out, n], grid_size=1, threadgroup_size=256)
        return None

    result = _measure(setup, run, "gpu_kernel_softmax")
    result.update(id="gpu_kernel_softmax", axis="gpu",
                  params={"n": n}, warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


def scenario_gpu_kernel_rmsnorm():
    """GPU MSL rmsnorm kernel."""
    rows, hidden = 2, 64

    def setup():
        ctx = MetalContext()
        inp = MetalTensor.empty(ctx, [rows, hidden])
        w = MetalTensor.empty(ctx, [hidden])
        out = MetalTensor.empty(ctx, [rows, hidden])
        return ctx, inp, w, out

    def run(ctx_tuple):
        ctx, inp, w, out = ctx_tuple
        ctx.dispatch_kernel("rmsnorm", [inp, out, w, hidden, 1e-6],
                            grid_size=rows * 256, threadgroup_size=256)
        return None

    result = _measure(setup, run, "gpu_kernel_rmsnorm")
    result.update(id="gpu_kernel_rmsnorm", axis="gpu",
                  params={"rows": rows, "hidden_dim": hidden},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    return result


# --- GPU Forward/Train ---

def _gpu_forward_scenario(label, hidden_dim, cfg_method):
    """Full-model GPU forward at given scale."""
    batch, seq = 2, 32

    def setup():
        gpu_seed_rng(SEED)
        cfg = cfg_method()
        ctx = MetalContext()
        model = MetalMoETransformer.from_config(ctx, cfg)
        token_ids = (np.arange(batch * seq, dtype=np.float32) % VOCAB).reshape(batch, seq)
        token_ids_mt = MetalTensor(ctx, data=token_ids)
        return model, token_ids_mt, batch, seq

    def run(ctx_tuple):
        model, token_ids_mt, b, s = ctx_tuple
        model.forward_tokens_tensor(token_ids_mt, b, s)
        return None

    result = _measure(setup, run, f"gpu_forward_{label}")
    result.update(id=f"gpu_forward_{label}", axis="gpu",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": hidden_dim},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def _gpu_train_scenario(label, hidden_dim, cfg_method):
    """Full-model GPU train proxy at given scale (forward + CE loss sum)."""
    batch, seq = 2, 8

    def setup():
        gpu_seed_rng(SEED)
        cfg = cfg_method()
        ctx = MetalContext()
        model = MetalMoETransformer.from_config(ctx, cfg)
        trainer = MetalTrainer(model)
        token_ids = (np.arange(batch * seq, dtype=np.float32) % VOCAB).reshape(batch, seq)
        targets = ((np.arange(batch * seq, dtype=np.float32) + 1.0) % VOCAB)
        token_ids_mt = MetalTensor(ctx, data=token_ids)
        targets_mt = MetalTensor(ctx, data=targets)
        return trainer, token_ids_mt, targets_mt, batch, seq

    def run(ctx_tuple):
        trainer, token_ids_mt, targets_mt, b, s = ctx_tuple
        trainer.train_step_gpu_tensors(
            token_ids_mt,
            targets_mt,
            batch=b,
            seq_len=s,
            readback=False,
        )
        return None

    result = _measure(setup, run, f"gpu_train_{label}")
    result.update(id=f"gpu_train_{label}", axis="gpu",
                  params={"batch": batch, "seq_len": seq, "hidden_dim": hidden_dim},
                  warmup_runs=N_WARMUP, trial_runs=N_TRIALS)
    _add_throughput(result, batch, seq)
    return result


def _metadata():
    return {
        "language": "python",
        "backend": "gpu_metal",
        "language_version": platform.python_version(),
        "os": platform.system() + " " + platform.release(),
        "cpu_model": platform.processor(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_trials": N_TRIALS,
        "n_warmup": N_WARMUP,
    }


def main():
    _log("Python MoE Benchmark -- GPU-only Metal (no host transfer in timed loop)")
    np.random.seed(SEED)

    scenarios = []
    total = 9
    idx = 0

    # Kernel benchmarks
    idx += 1
    _log(f"[{idx}/{total}] gpu_kernel_matmul")
    scenarios.append(scenario_gpu_kernel_matmul())

    idx += 1
    _log(f"[{idx}/{total}] gpu_kernel_softmax")
    scenarios.append(scenario_gpu_kernel_softmax())

    idx += 1
    _log(f"[{idx}/{total}] gpu_kernel_rmsnorm")
    scenarios.append(scenario_gpu_kernel_rmsnorm())

    # Forward pass
    idx += 1
    _log(f"[{idx}/{total}] gpu_forward_64")
    scenarios.append(_gpu_forward_scenario("64", 64, Config.tiny))

    idx += 1
    _log(f"[{idx}/{total}] gpu_forward_256")
    scenarios.append(_gpu_forward_scenario("256", 256, Config.small))

    idx += 1
    _log(f"[{idx}/{total}] gpu_forward_512")
    scenarios.append(_gpu_forward_scenario("512", 512, Config.medium))

    # Train step
    idx += 1
    _log(f"[{idx}/{total}] gpu_train_64")
    scenarios.append(_gpu_train_scenario("64", 64, Config.tiny))

    idx += 1
    _log(f"[{idx}/{total}] gpu_train_256")
    scenarios.append(_gpu_train_scenario("256", 256, Config.small))

    idx += 1
    _log(f"[{idx}/{total}] gpu_train_512")
    scenarios.append(_gpu_train_scenario("512", 512, Config.medium))

    result = {"metadata": _metadata(), "scenarios": scenarios}
    _log("Done. Writing JSON to stdout.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
