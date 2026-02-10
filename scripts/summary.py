#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Print benchmark summary tables from benchmarks/*.json."""

import json
import os
import sys

FLOPS_MATMUL_64 = 2 * 64 * 64 * 64  # 524288
FLOPS_SOFTMAX = 4 * 1000  # 4 ops * n
FLOPS_RMSNORM = 3 * 2 * 32 * 64  # 3 ops * shape


def load_all(root):
    langs = ["rust", "julia", "go", "python"]
    out = {}
    for lang in langs:
        path = os.path.join(root, f"{lang}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        out[lang] = {s["id"]: s for s in data["scenarios"]}
    return out


def ns_to_ms(ns):
    return ns / 1e6


def ns_to_us(ns):
    return ns / 1e3


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
        fwd = ns_to_ms(sc.get("dispatch_warm", {}).get("median_ns", 0))
        trn = ns_to_ms(sc.get("mem_train_step", {}).get("median_ns", 0))
        mm_ns = sc.get("kernel_matmul", {}).get("median_ns", 1)
        gf = FLOPS_MATMUL_64 / (mm_ns / 1e9) / 1e9 if mm_ns > 0 else 0
        t4_ns = sc.get("parallel_T4", {}).get("median_ns", 0)
        t4 = 4.0 / (t4_ns / 1e9) if t4_ns > 0 else 0
        rss = sc.get("dispatch_warm", {}).get("memory", {}).get(
            "peak_rss_bytes", 0) / 1e6
        print(f"{lang.capitalize():>8s}"
              f" {fwd:7.2f} ms"
              f" {trn:7.2f} ms"
              f" {gf:6.0f} GF"
              f" {t4:7.0f} inf/s"
              f" {rss:5.0f} MB")

    # --- Table 2: Kernel throughput ---
    print(f"\n=== Kernel Throughput ===")
    print(f"{'Language':>8s} {'softmax':>12s} {'rmsnorm':>12s}"
          f" {'matmul':>12s}")
    print("-" * 48)
    for lang, sc in all_data.items():
        soft = ns_to_us(sc.get("kernel_softmax", {}).get("median_ns", 0))
        rms = ns_to_us(sc.get("kernel_rmsnorm", {}).get("median_ns", 0))
        mm = ns_to_us(sc.get("kernel_matmul", {}).get("median_ns", 0))
        print(f"{lang.capitalize():>8s}"
              f" {soft:8.2f} us"
              f" {rms:8.2f} us"
              f" {mm:8.2f} us")

    # --- Table 3: Parallel scaling ---
    print(f"\n=== Parallel Scaling ===")
    print(f"{'Language':>8s} {'T1':>12s} {'T2':>12s}"
          f" {'T4':>12s} {'T4/T1':>8s}")
    print("-" * 56)
    for lang, sc in all_data.items():
        vals = {}
        for t in ["T1", "T2", "T4"]:
            ns = sc.get(f"parallel_{t}", {}).get("median_ns", 0)
            n_threads = int(t[1])
            vals[t] = n_threads / (ns / 1e9) if ns > 0 else 0
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
        for t in ["T1", "T2", "T4"]:
            ns = sc.get(f"parallel_train_{t}", {}).get("median_ns", 0)
            n_threads = int(t[1])
            vals[t] = n_threads / (ns / 1e9) if ns > 0 else 0
        speedup = vals["T4"] / vals["T1"] if vals["T1"] > 0 else 0
        print(f"{lang.capitalize():>8s}"
              f" {vals['T1']:7.0f} trn/s"
              f" {vals['T2']:7.0f} trn/s"
              f" {vals['T4']:7.0f} trn/s"
              f" {speedup:5.2f}x")

    # --- Table 4: Type system dispatch ---
    print(f"\n=== Type System Dispatch ===")
    print(f"{'Language':>8s} {'Warm':>10s} {'Cold':>10s} {'Ratio':>8s}")
    print("-" * 40)
    for lang, sc in all_data.items():
        warm = ns_to_ms(sc.get("dispatch_warm", {}).get("median_ns", 0))
        cold = ns_to_ms(sc.get("dispatch_cold", {}).get("median_ns", 0))
        ratio = cold / warm if warm > 0 else 0
        print(f"{lang.capitalize():>8s}"
              f" {warm:7.2f} ms"
              f" {cold:7.2f} ms"
              f" {ratio:5.2f}x")

    # --- Table 5: Memory & GC ---
    print(f"\n=== Memory & GC (train step) ===")
    print(f"{'Language':>8s} {'alloc':>10s} {'GC time':>10s}"
          f" {'gc_tput':>8s} {'RSS':>8s}")
    print("-" * 50)
    for lang, sc in all_data.items():
        trn = sc.get("mem_train_step", {})
        mem = trn.get("memory", {})
        alloc = mem.get("alloc_bytes", 0) or 0
        gc_ns = mem.get("gc_time_ns", 0) or 0
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

    # --- Spreads ---
    print(f"\n=== Spreads ===")
    for label, scenario_id in [("Forward h=64", "dispatch_warm"),
                                ("Train h=64", "mem_train_step"),
                                ("Forward h=256", "scale_forward_256"),
                                ("Train h=256", "scale_train_256")]:
        times = {}
        for lang, sc in all_data.items():
            s = sc.get(scenario_id)
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
