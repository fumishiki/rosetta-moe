#!/bin/bash
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer
#
# Verify CPU/GPU separation: no cross-boundary imports or data transfers.
# Exit 0 if clean, exit 1 if violations found.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ERRORS=0

ok()  { printf "  \033[32m✓\033[0m %s\n" "$1"; }
err() { printf "  \033[31m✗\033[0m %s\n" "$1"; ERRORS=$((ERRORS + 1)); }

check_absent() {
    local pattern="$1"
    local dir="$2"
    local label="$3"
    # Filter out: __pycache__, comments (// # """ '''), and doc strings
    local matches
    matches=$(grep -rn "$pattern" "$dir" 2>/dev/null \
        | grep -v '__pycache__' \
        | grep -v '^\([^:]*:[0-9]*:\s*//\)' \
        | grep -v '^\([^:]*:[0-9]*:\s*#\)' \
        | grep -v '^\([^:]*:[0-9]*:.*//!\)' \
        | grep -vE '^[^:]+:[0-9]+:\s*("""|'"'''"')' \
        | grep -vE '^[^:]+:[0-9]+:("""|'"'''"')' \
        | grep -vE '^[^:]+:[0-9]+:[A-Z].*\. No ' \
        || true)
    if [ -n "$matches" ]; then
        echo "$matches"
        err "$label"
    else
        ok "$label"
    fi
}

# ============================================================
# Rust: rust/src/gpu/ must not reference cpu module or Vec<f32>
# ============================================================
printf "\033[1;36m[Rust] Checking GPU isolation...\033[0m\n"

check_absent 'use.*::cpu::' \
    "$ROOT/rust/src/gpu/" \
    "No cpu:: imports in gpu/"

check_absent 'Vec<f32>' \
    "$ROOT/rust/src/gpu/metal_tensor.rs" \
    "No Vec<f32> in metal_tensor.rs"

check_absent 'to_vec()' \
    "$ROOT/rust/src/gpu/" \
    "No to_vec() in gpu/"

check_absent 'from_slice(' \
    "$ROOT/rust/src/gpu/" \
    "No from_slice() in gpu/"

check_absent 'contents_ptr(' \
    "$ROOT/rust/src/gpu/metal_model.rs" \
    "No host pointer reads in metal_model.rs"

check_absent 'gpu_train_step\(' \
    "$ROOT/rust/benches/bench.rs" \
    "Rust GPU bench avoids scalar-readback train API"

check_absent 'read_f32\(' \
    "$ROOT/rust/benches/bench.rs" \
    "No read_f32() in Rust GPU bench"

# Rust CPU must not reference gpu module
printf "\033[1;36m[Rust] Checking CPU isolation...\033[0m\n"

check_absent 'use.*::gpu::' \
    "$ROOT/rust/src/cpu/" \
    "No gpu:: imports in cpu/"

check_absent 'metal' \
    "$ROOT/rust/src/cpu/" \
    "No metal references in cpu/"

# ============================================================
# Julia: src_gpu/ must not use Array(MtlArray); src_cpu/ must not use MtlArray
# ============================================================
printf "\033[1;36m[Julia] Checking GPU isolation...\033[0m\n"

check_absent 'Array(' \
    "$ROOT/julia/src_gpu/" \
    "No Array() conversion in src_gpu/"

check_absent 'collect(' \
    "$ROOT/julia/src_gpu/" \
    "No collect() in src_gpu/"

check_absent 'gpu_train_step!\(' \
    "$ROOT/julia/bench_gpu.jl" \
    "Julia GPU bench avoids scalar-readback train API"

printf "\033[1;36m[Julia] Checking CPU isolation...\033[0m\n"

check_absent 'MtlArray' \
    "$ROOT/julia/src_cpu/" \
    "No MtlArray in src_cpu/"

check_absent 'Metal\.' \
    "$ROOT/julia/src_cpu/" \
    "No Metal. references in src_cpu/"

# ============================================================
# Python: gpu/ must not use numpy(); cpu/ must not import Metal
# ============================================================
printf "\033[1;36m[Python] Checking GPU isolation...\033[0m\n"

check_absent '\.numpy()' \
    "$ROOT/python/gpu/" \
    "No .numpy() in gpu/"

check_absent '\.cpu()' \
    "$ROOT/python/gpu/" \
    "No .cpu() in gpu/"

check_absent 'numpy_view\(' \
    "$ROOT/python/gpu/metal_attention.py" \
    "No numpy_view() in python GPU attention path"

check_absent 'numpy_view\(' \
    "$ROOT/python/gpu/metal_moe.py" \
    "No numpy_view() in python GPU MoE path"

printf "\033[1;36m[Python] Checking CPU isolation...\033[0m\n"

check_absent 'import.*metal' \
    "$ROOT/python/cpu/" \
    "No metal imports in cpu/"

check_absent 'import.*Metal\|from.*Metal\|Metal\.' \
    "$ROOT/python/cpu/" \
    "No Metal references in cpu/"

# ============================================================
# Go: gpu/ must not import cpu package; cpu/ must not import gpu
# ============================================================
printf "\033[1;36m[Go] Checking GPU isolation...\033[0m\n"

check_absent '".*\/cpu"' \
    "$ROOT/go/gpu/" \
    "No cpu package imports in gpu/"

check_absent 'copy(' \
    "$ROOT/go/gpu/metal.go" \
    "No slice copy() in gpu/metal.go"

check_absent '\.Data\(' \
    "$ROOT/go/gpu/bench_gpu_test.go" \
    "No host Data() reads in Go GPU bench"

printf "\033[1;36m[Go] Checking CPU isolation...\033[0m\n"

check_absent '".*\/gpu"' \
    "$ROOT/go/cpu/" \
    "No gpu package imports in cpu/"

check_absent 'metal' \
    "$ROOT/go/cpu/" \
    "No metal references in cpu/"

# ============================================================
# Summary
# ============================================================
echo ""
if [ "$ERRORS" -gt 0 ]; then
    printf "\033[31m✗ %d separation violation(s) found.\033[0m\n" "$ERRORS"
    exit 1
else
    printf "\033[32m✓ All CPU/GPU separation checks passed.\033[0m\n"
    exit 0
fi
