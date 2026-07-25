# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# bench.jl — Compatibility dispatcher for separated CPU/GPU benchmarks
#
# Policy:
# - CPU benchmark: julia/bench_cpu.jl
# - GPU benchmark: julia/bench_gpu.jl
#
# Environment switches:
#   ROSETTA_CPU_ONLY=1 -> bench_cpu.jl
#   ROSETTA_GPU_ONLY=1 -> bench_gpu.jl
# If neither is set, defaults to CPU benchmark.

cpu_only = get(ENV, "ROSETTA_CPU_ONLY", "") == "1"
gpu_only = get(ENV, "ROSETTA_GPU_ONLY", "") == "1"

if cpu_only && gpu_only
    error("ROSETTA_CPU_ONLY and ROSETTA_GPU_ONLY cannot both be set")
elseif gpu_only
    include("bench_gpu.jl")
else
    include("bench_cpu.jl")
end
