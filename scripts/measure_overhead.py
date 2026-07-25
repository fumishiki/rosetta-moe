#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Measure language overhead by comparing step time vs kernel time.

Language overhead = Step time - Kernel time
This captures dispatch, type system, memory management, and GC costs
that each language adds on top of the raw compute kernels.

Reads benchmark JSON files from benchmarks/ directory.
"""

import json
import os
import sys


def ns_to_ms(ns):
    return ns / 1e6


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data["scenarios"]}


def extract_kernel_time(scenarios):
    """Sum of kernel micro-benchmark medians (matmul + softmax + rmsnorm)."""
    total = 0
    for kid in ["kernel_matmul", "kernel_softmax", "kernel_rmsnorm"]:
        s = scenarios.get(kid)
        if s:
            total += s.get("median_ns", 0)
    return total


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    langs = ["rust", "julia", "go", "python"]

    cpu_data = {}
    gpu_data = {}
    for lang in langs:
        cpu = load_json(os.path.join(root, f"{lang}.json"))
        if cpu:
            cpu_data[lang] = cpu
        gpu = load_json(os.path.join(root, f"{lang}_gpu.json"))
        if gpu:
            gpu_data[lang] = gpu

    if not cpu_data and not gpu_data:
        print("No benchmark JSON files found. Run 'make bench-cpu' or 'make bench-gpu' first.")
        sys.exit(1)

    # --- CPU overhead table ---
    if cpu_data:
        print("=== CPU Language Overhead (h=64) ===")
        print(f"{'Language':>8s} {'Step':>10s} {'Kernels':>10s} {'Overhead':>10s} {'Overhead%':>10s}")
        print("-" * 52)
        for lang in langs:
            sc = cpu_data.get(lang)
            if not sc:
                continue
            step_ns = sc.get("dispatch_warm", {}).get("median_ns", 0)
            kernel_ns = extract_kernel_time(sc)
            overhead_ns = step_ns - kernel_ns
            pct = (overhead_ns / step_ns * 100) if step_ns > 0 else 0
            print(f"{lang.capitalize():>8s}"
                  f" {ns_to_ms(step_ns):7.2f} ms"
                  f" {ns_to_ms(kernel_ns):7.2f} ms"
                  f" {ns_to_ms(overhead_ns):7.2f} ms"
                  f" {pct:7.1f}%")

    # --- GPU overhead table ---
    if gpu_data:
        print(f"\n=== GPU Language Overhead (h=64) ===")
        print(f"{'Language':>8s} {'Step':>10s} {'MPS/Metal':>10s} {'Overhead':>10s} {'Overhead%':>10s}")
        print("-" * 52)
        for lang in langs:
            sc = gpu_data.get(lang)
            if not sc:
                continue
            step_ns = sc.get("gpu_forward_64", {}).get("median_ns", 0)
            kernel_ns = extract_kernel_time(sc)
            overhead_ns = step_ns - kernel_ns
            pct = (overhead_ns / step_ns * 100) if step_ns > 0 else 0
            print(f"{lang.capitalize():>8s}"
                  f" {ns_to_ms(step_ns):7.2f} ms"
                  f" {ns_to_ms(kernel_ns):7.2f} ms"
                  f" {ns_to_ms(overhead_ns):7.2f} ms"
                  f" {pct:7.1f}%")

    # --- Training overhead table ---
    if cpu_data:
        print(f"\n=== CPU Training Overhead (h=64) ===")
        print(f"{'Language':>8s} {'Train Step':>12s} {'Kernels':>10s} {'Overhead':>10s} {'Overhead%':>10s}")
        print("-" * 56)
        for lang in langs:
            sc = cpu_data.get(lang)
            if not sc:
                continue
            step_ns = sc.get("mem_train_step", {}).get("median_ns", 0)
            kernel_ns = extract_kernel_time(sc)
            overhead_ns = step_ns - kernel_ns
            pct = (overhead_ns / step_ns * 100) if step_ns > 0 else 0
            print(f"{lang.capitalize():>8s}"
                  f" {ns_to_ms(step_ns):9.2f} ms"
                  f" {ns_to_ms(kernel_ns):7.2f} ms"
                  f" {ns_to_ms(overhead_ns):7.2f} ms"
                  f" {pct:7.1f}%")

    # --- Overhead ranking ---
    if cpu_data:
        print(f"\n=== Overhead Ranking (lowest = best) ===")
        ranked = []
        for lang in langs:
            sc = cpu_data.get(lang)
            if not sc:
                continue
            step_ns = sc.get("dispatch_warm", {}).get("median_ns", 0)
            kernel_ns = extract_kernel_time(sc)
            if step_ns > 0:
                pct = (step_ns - kernel_ns) / step_ns * 100
                ranked.append((lang, pct))
        ranked.sort(key=lambda x: x[1])
        for i, (lang, pct) in enumerate(ranked, 1):
            print(f"  {i}. {lang.capitalize():>8s}: {pct:5.1f}% overhead")


if __name__ == "__main__":
    main()
