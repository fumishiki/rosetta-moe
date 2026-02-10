#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

"""Verify docs/bench-results.md and README.md against benchmarks/*.json.

Reads JSON benchmark outputs, computes expected metric values, and checks
that each value appears in the corresponding documentation file.
Designed to run AFTER 'make bench' regenerates fresh JSON.

Usage:
    python3 scripts/check_docs.py          # normal
    python3 scripts/check_docs.py -v       # verbose (show passes too)
    make check-docs                        # via Makefile
"""

import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RTOL = 0.07  # 7% relative tolerance for doc rounding
FLOPS_MATMUL_64 = 2 * 64 * 64 * 64


def load_scenarios():
    data = {}
    for lang in ("rust", "julia", "go", "python"):
        path = os.path.join(ROOT, "benchmarks", f"{lang}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data[lang] = {s["id"]: s for s in json.load(f)["scenarios"]}
    return data


def load_doc(relpath):
    full = os.path.join(ROOT, relpath)
    if not os.path.exists(full):
        return ""
    with open(full) as f:
        return f.read()


def extract_section(doc, heading_pattern, level=3):
    """Extract text between a heading and the next heading of same/higher level."""
    pat = rf"^{'#' * level}\s+{heading_pattern}.*$"
    m = re.search(pat, doc, re.MULTILINE | re.IGNORECASE)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(rf"^#{{1,{level}}}\s+", doc[start:], re.MULTILINE)
    if nxt:
        return doc[start:start + nxt.start()]
    return doc[start:]


def numbers_in_text(text):
    """Extract all decimal/integer numbers from text."""
    return [float(m.replace(",", "")) for m in
            re.findall(r"\d{1,3}(?:,\d{3})+\.?\d*|\d+\.\d+|\d+", text)]


def value_in_text(expected, text, tol=RTOL):
    """Check if expected value appears approximately in text."""
    for n in numbers_in_text(text):
        if expected == 0 and n == 0:
            return True
        if expected != 0 and abs(n - expected) / abs(expected) <= tol:
            return True
    return False


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    data = load_scenarios()

    if not data:
        print("No benchmark JSON found. Run 'make bench' first.")
        return 1

    bench = load_doc("docs/bench-results.md")
    readme = load_doc("README.md")

    if not bench:
        print("docs/bench-results.md not found.")
        return 1

    errors = 0
    passed = 0

    def check(doc_name, section_text, expected, label):
        nonlocal errors, passed
        if not section_text:
            if verbose:
                print(f"  SKIP [{doc_name}] {label}: section not found")
            return
        if value_in_text(expected, section_text):
            passed += 1
            if verbose:
                print(f"  OK   [{doc_name}] {label} = {expected:.2f}")
        else:
            print(f"  FAIL [{doc_name}] {label}: expected ~{expected:.2f}")
            errors += 1

    # --- Extract sections from bench-results.md ---
    sec_train = extract_section(bench, r"Axis 1.*Memory", level=3)
    sec_kernel = extract_section(bench, r"Axis 2.*Compiler", level=3)
    sec_dispatch = extract_section(bench, r"Axis 3.*Type", level=3)
    sec_parallel = extract_section(bench, r"Axis 4.*Parallel", level=3)
    sec_scale = extract_section(bench, r"Scaling Analysis", level=3)
    sec_rankings = extract_section(bench, r"Rankings", level=3)

    # --- h=64 forward (dispatch_warm) ---
    print("--- h=64 Forward (dispatch_warm) ---")
    for lang, sc in data.items():
        s = sc.get("dispatch_warm")
        if s:
            ms = s["median_ns"] / 1e6
            check("bench/dispatch", sec_dispatch, ms,
                  f"{lang} warm {ms:.2f}ms")

    # --- h=64 training (mem_train_step) ---
    print("--- h=64 Training (mem_train_step) ---")
    for lang, sc in data.items():
        s = sc.get("mem_train_step")
        if s:
            ms = s["median_ns"] / 1e6
            check("bench/memory", sec_train, ms,
                  f"{lang} train {ms:.2f}ms")

    # --- Kernel softmax/rmsnorm ---
    print("--- Kernels ---")
    for lang, sc in data.items():
        for kid in ("kernel_softmax", "kernel_rmsnorm"):
            s = sc.get(kid)
            if s:
                us = s["median_ns"] / 1e3
                check("bench/kernel", sec_kernel, us,
                      f"{lang} {kid} {us:.2f}us")

    # --- Kernel matmul GFLOPS ---
    for lang, sc in data.items():
        s = sc.get("kernel_matmul")
        if s:
            gf = FLOPS_MATMUL_64 / (s["median_ns"] / 1e9) / 1e9
            check("bench/kernel", sec_kernel, gf,
                  f"{lang} matmul {gf:.0f} GF")

    # --- Parallel T4 throughput ---
    print("--- Parallel T4 ---")
    for lang, sc in data.items():
        s = sc.get("parallel_T4")
        if s:
            infs = 4.0 / (s["median_ns"] / 1e9)
            check("bench/parallel", sec_parallel, infs,
                  f"{lang} T4 {infs:.0f} inf/s")

    # --- Parallel Training T4 throughput ---
    print("--- Parallel Training T4 ---")
    for lang, sc in data.items():
        s = sc.get("parallel_train_T4")
        if s:
            trns = 4.0 / (s["median_ns"] / 1e9)
            check("bench/parallel", sec_parallel, trns,
                  f"{lang} T4 train {trns:.0f} trn/s")

    # --- h=256 scale ---
    print("--- h=256 Scale ---")
    for lang, sc in data.items():
        for sid, tag in [("scale_forward_256", "fwd"),
                         ("scale_train_256", "trn")]:
            s = sc.get(sid)
            if s:
                ms = s["median_ns"] / 1e6
                check("bench/scale", sec_scale, ms,
                      f"{lang} {tag} h=256 {ms:.2f}ms")
                if readme:
                    check("README", readme, ms,
                          f"{lang} {tag} h=256 {ms:.2f}ms")

    # --- Forward spread ---
    print("--- Spreads ---")
    for label, sid in [("fwd h=64", "dispatch_warm"),
                       ("train h=64", "mem_train_step"),
                       ("fwd h=256", "scale_forward_256"),
                       ("train h=256", "scale_train_256")]:
        times = {}
        for lang, sc in data.items():
            s = sc.get(sid)
            if s:
                times[lang] = s["median_ns"]
        if times:
            spread = max(times.values()) / min(times.values())
            check("bench/scale", sec_scale if "256" in label else bench,
                  spread, f"{label} spread {spread:.2f}x")

    # --- Rankings table ---
    print("--- Rankings ---")
    for lang, sc in data.items():
        s = sc.get("scale_forward_256")
        if s:
            ms = s["median_ns"] / 1e6
            check("bench/rankings", sec_rankings, ms,
                  f"{lang} scale fwd h=256 {ms:.2f}ms in rankings")

    # --- Summary ---
    print(f"\n{'='*40}")
    print(f"Passed: {passed}  Failed: {errors}")
    if errors:
        print("\nHint: Run 'make bench' to regenerate JSON, then re-check.")
        print("JSON files may be stale from a previous benchmark run.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
