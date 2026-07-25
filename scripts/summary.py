#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Print benchmark summary tables from benchmarks/*.json."""

import json
import os
import sys

FLOPS_MATMUL_64 = 2 * 64 * 64 * 64  # 524288
FLOPS_SOFTMAX = 4 * 1000  # 4 ops * n
FLOPS_RMSNORM = 3 * 2 * 32 * 64  # 3 ops * shape

CPU_ID_ALIASES = {
    "dispatch_warm": ("dispatch_warm", "cpu_forward"),
    "mem_train_step": ("mem_train_step", "cpu_train_step"),
    "kernel_matmul": ("kernel_matmul", "cpu_kernel_matmul"),
    "kernel_softmax": ("kernel_softmax", "cpu_kernel_softmax"),
    "kernel_rmsnorm": ("kernel_rmsnorm", "cpu_kernel_rmsnorm"),
    "parallel_T1": ("parallel_T1", "cpu_parallel_T1"),
    "parallel_T2": ("parallel_T2", "cpu_parallel_T2"),
    "parallel_T4": ("parallel_T4", "cpu_parallel_T4"),
}


def load_all(root):
    langs = ["rust", "julia", "go", "python"]
    out = {}
    for lang in langs:
        path = os.path.join(root, f"{lang}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            out[lang] = {s["id"]: s for s in data["scenarios"]}

        # Also load GPU results if available
        gpu_path = os.path.join(root, f"{lang}_gpu.json")
        if os.path.exists(gpu_path):
            with open(gpu_path) as f:
                gpu_data = json.load(f)
            if lang not in out:
                out[lang] = {}
            for s in gpu_data["scenarios"]:
                out[lang][s["id"]] = s
    return out


def ns_to_ms(ns):
    return ns / 1e6


def ns_to_us(ns):
    return ns / 1e3


def scenario_by_id(sc, scenario_id):
    aliases = CPU_ID_ALIASES.get(scenario_id, (scenario_id,))
    for sid in aliases:
        s = sc.get(sid)
        if s:
            return s
    return None


def median_ns_by_id(sc, scenario_id):
    s = scenario_by_id(sc, scenario_id)
    return s.get("median_ns") if s else None


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    all_data = load_all(root)

    if not all_data:
        print("No benchmark JSON files found. Run 'make bench' first.")
        sys.exit(1)

    # --- Table 1: h=64 core metrics ---
    print("=== h=64 Core Metrics ===")
    print(f"{'Language':>8s} {'Forward':>10s} {'Train':>10s} {'Matmul':>10s}"
          f" {'T4':>12s} {'RSS':>8s}")
    print("-" * 62)
    for lang, sc in all_data.items():
        fwd_s = scenario_by_id(sc, "dispatch_warm")
        trn_s = scenario_by_id(sc, "mem_train_step")
        mm_s = scenario_by_id(sc, "kernel_matmul")
        t4_s = scenario_by_id(sc, "parallel_T4")

        fwd_ns = fwd_s.get("median_ns") if fwd_s else None
        trn_ns = trn_s.get("median_ns") if trn_s else None
        mm_ns = mm_s.get("median_ns") if mm_s else None
        t4_ns = t4_s.get("median_ns") if t4_s else None

        fwd_str = f"{ns_to_ms(fwd_ns):7.2f} ms" if fwd_ns else "    N/A"
        trn_str = f"{ns_to_ms(trn_ns):7.2f} ms" if trn_ns else "    N/A"
        gf_str = (
            f"{FLOPS_MATMUL_64 / (mm_ns / 1e9) / 1e9:6.0f} GF"
            if mm_ns and mm_ns > 0 else "    N/A"
        )
        t4 = 4.0 / (t4_ns / 1e9) if t4_ns and t4_ns > 0 else 0
        t4_str = f"{t4:7.0f} inf/s" if t4_ns else "     N/A"
        rss = (fwd_s or trn_s or {}).get("memory", {}).get("peak_rss_bytes", 0) / 1e6
        print(f"{lang.capitalize():>8s}"
              f" {fwd_str}"
              f" {trn_str}"
              f" {gf_str}"
              f" {t4_str}"
              f" {rss:5.0f} MB")

    # --- Table 2: Kernel throughput ---
    print(f"\n=== Kernel Throughput ===")
    print(f"{'Language':>8s} {'softmax':>12s} {'rmsnorm':>12s}"
          f" {'matmul':>12s}")
    print("-" * 48)
    for lang, sc in all_data.items():
        soft_ns = median_ns_by_id(sc, "kernel_softmax")
        rms_ns = median_ns_by_id(sc, "kernel_rmsnorm")
        mm_ns = median_ns_by_id(sc, "kernel_matmul")
        soft = f"{ns_to_us(soft_ns):8.2f} us" if soft_ns else "     N/A"
        rms = f"{ns_to_us(rms_ns):8.2f} us" if rms_ns else "     N/A"
        mm = f"{ns_to_us(mm_ns):8.2f} us" if mm_ns else "     N/A"
        print(f"{lang.capitalize():>8s}"
              f" {soft}"
              f" {rms}"
              f" {mm}")

    # --- Table 3: Parallel scaling ---
    print(f"\n=== Parallel Scaling ===")
    print(f"{'Language':>8s} {'T1':>12s} {'T2':>12s}"
          f" {'T4':>12s} {'T4/T1':>8s}")
    print("-" * 56)
    for lang, sc in all_data.items():
        vals = {}
        for t in ["T1", "T2", "T4"]:
            ns = median_ns_by_id(sc, f"parallel_{t}")
            n_threads = int(t[1])
            vals[t] = n_threads / (ns / 1e9) if ns and ns > 0 else 0
        speedup = vals["T4"] / vals["T1"] if vals["T1"] > 0 else 0
        print(f"{lang.capitalize():>8s}"
              f" {vals['T1']:7.0f} inf/s"
              f" {vals['T2']:7.0f} inf/s"
              f" {vals['T4']:7.0f} inf/s"
              f" {speedup:5.2f}x")

    # --- Table 3b: Parallel Training Scaling ---
    print(f"\n=== Parallel Training Scaling ===")
    print(f"{'Language':>8s} {'T1':>12s} {'T2':>12s}"
          f" {'T4':>12s} {'T4/T1':>8s}")
    print("-" * 56)
    for lang, sc in all_data.items():
        vals = {}
        strs = {}
        for t in ["T1", "T2", "T4"]:
            ns = sc.get(f"parallel_train_{t}", {}).get("median_ns")
            n_threads = int(t[1])
            if ns and ns > 0:
                vals[t] = n_threads / (ns / 1e9)
                strs[t] = f"{vals[t]:7.0f} trn/s"
            else:
                vals[t] = 0
                strs[t] = "    N/A"
        speedup = vals["T4"] / vals["T1"] if vals["T1"] > 0 else 0
        speedup_str = f"{speedup:5.2f}x" if vals["T1"] > 0 else "  N/A"
        print(f"{lang.capitalize():>8s}"
              f" {strs['T1']}"
              f" {strs['T2']}"
              f" {strs['T4']}"
              f" {speedup_str}")

    # --- Table 4: Type system dispatch ---
    print(f"\n=== Type System Dispatch ===")
    print(f"{'Language':>8s} {'Warm':>10s} {'Cold':>10s} {'Ratio':>8s}")
    print("-" * 40)
    for lang, sc in all_data.items():
        warm_ns = median_ns_by_id(sc, "dispatch_warm")
        cold_ns = sc.get("dispatch_cold", {}).get("median_ns")
        warm = ns_to_ms(warm_ns) if warm_ns else 0
        cold = ns_to_ms(cold_ns) if cold_ns else 0
        ratio = cold / warm if warm > 0 and cold_ns else 0
        cold_str = f"{cold:7.2f} ms" if cold_ns else "    N/A"
        ratio_str = f"{ratio:5.2f}x" if cold_ns and warm > 0 else "  N/A"
        print(f"{lang.capitalize():>8s}"
              f" {warm:7.2f} ms"
              f" {cold_str}"
              f" {ratio_str}")

    # --- Table 5: Memory & GC ---
    print(f"\n=== Memory & GC (train step) ===")
    print(f"{'Language':>8s} {'alloc':>10s} {'GC time':>10s}"
          f" {'gc_tput':>8s} {'RSS':>8s}")
    print("-" * 50)
    for lang, sc in all_data.items():
        trn = scenario_by_id(sc, "mem_train_step") or {}
        mem = trn.get("memory", {})
        alloc = mem.get("alloc_bytes", 0) or 0
        gc_ns = trn.get("gc", {}).get("total_gc_time_ns", mem.get("gc_time_ns", 0)) or 0
        med = trn.get("median_ns", 1)
        gc_tp = 1.0 - (gc_ns / med) if med > 0 else 1.0
        rss = mem.get("peak_rss_bytes", 0) or 0
        print(f"{lang.capitalize():>8s}"
              f" {alloc/1e6:6.1f} MB"
              f" {gc_ns/1e6:7.2f} ms"
              f" {gc_tp:7.3f}"
              f" {rss/1e6:5.0f} MB")

    # --- Table 6: h=256 scale ---
    print(f"\n=== h=256 Scale Comparison ===")
    print(f"{'Language':>8s} {'Fwd h=256':>10s} {'Train h=256':>12s}")
    print("-" * 34)
    has_scale = False
    for lang, sc in all_data.items():
        sf = sc.get("scale_forward_256")
        st = sc.get("scale_train_256")
        if sf and st:
            has_scale = True
            print(f"{lang.capitalize():>8s}"
                  f" {ns_to_ms(sf['median_ns']):7.2f} ms"
                  f" {ns_to_ms(st['median_ns']):9.2f} ms")
    if not has_scale:
        print("  (no scale_*_256 scenarios found)")

    # --- Table 7: h=512 scale ---
    print(f"\n=== h=512 Scale Comparison ===")
    print(f"{'Language':>8s} {'Fwd h=512':>10s} {'Train h=512':>12s}")
    print("-" * 34)
    has_scale_512 = False
    for lang, sc in all_data.items():
        sf = sc.get("scale_forward_512")
        st = sc.get("scale_train_512")
        if sf and st:
            has_scale_512 = True
            print(f"{lang.capitalize():>8s}"
                  f" {ns_to_ms(sf['median_ns']):7.2f} ms"
                  f" {ns_to_ms(st['median_ns']):9.2f} ms")
    if not has_scale_512:
        print("  (no scale_*_512 scenarios found)")

    # --- Table 8: GPU Metrics ---
    print(f"\n=== Axis 6: GPU ===")
    gpu_scenarios = ["gpu_forward_64", "gpu_forward_256", "gpu_forward_512"]
    has_gpu = False
    for lang, sc in all_data.items():
        for gid in gpu_scenarios:
            if gid in sc:
                has_gpu = True
                break
    if has_gpu:
        print(f"{'Language':>8s} {'Fwd 64':>10s} {'Fwd 256':>10s} {'Fwd 512':>10s}")
        print("-" * 42)
        for lang, sc in all_data.items():
            vals = []
            for gid in gpu_scenarios:
                s = sc.get(gid)
                vals.append(f"{ns_to_ms(s['median_ns']):7.2f} ms" if s else "     N/A")
            print(f"{lang.capitalize():>8s} {'  '.join(vals)}")
    else:
        print("  (no GPU scenarios found — run with Metal backend)")

    # --- Spreads ---
    print(f"\n=== Spreads ===")
    for label, scenario_id in [("Forward h=64", "dispatch_warm"),
                                ("Train h=64", "mem_train_step"),
                                ("Forward h=256", "scale_forward_256"),
                                ("Train h=256", "scale_train_256"),
                                ("Forward h=512", "scale_forward_512"),
                                ("Train h=512", "scale_train_512")]:
        times = {}
        for lang, sc in all_data.items():
            s = scenario_by_id(sc, scenario_id)
            if s:
                times[lang] = s["median_ns"]
        if times:
            fastest = min(times, key=times.get)
            slowest = max(times, key=times.get)
            spread = times[slowest] / times[fastest]
            print(f"  {label:>16s}: {spread:.2f}x "
                  f"({slowest}/{fastest})")


if __name__ == "__main__":
    main()
