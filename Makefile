# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# rosetta-moe — 4-language MoE Transformer benchmark
# Usage:
#   make test          Run all tests (Rust + Go + Python + Julia)
#   make bench-cpu     Run CPU benchmarks and save JSON
#   make bench-gpu     Run GPU benchmarks and save JSON
#   make bench-all     Run all CPU/GPU benchmarks sequentially (8 runs)
#   make bench-all-30  Run all CPU/GPU benchmarks with fixed 30 trials
#   make convergence   Run loss convergence verification (all 4 languages)
#   make test-rust     Run Rust tests only
#   make bench-cpu-julia  Run Julia CPU benchmark only
#   make bench-gpu-julia  Run Julia GPU benchmark only
#   make clean         Remove build artifacts and caches
#   make verify        Full verification: test + bench + convergence + summary
#   make verify-separation  Check CPU/GPU code isolation (no cross-boundary imports)
#   make bench-overhead     Run benchmarks + language overhead analysis

.PHONY: test test-rust test-go test-python test-julia \
        bench-cpu bench-cpu-rust bench-cpu-go bench-cpu-python bench-cpu-julia bench-cool \
        bench-gpu bench-gpu-rust bench-gpu-go bench-gpu-python bench-gpu-julia \
        bench-all bench-all-30 \
        convergence convergence-rust convergence-go convergence-python convergence-julia convergence-plots \
        convergence-gpu convergence-gpu-rust convergence-gpu-go convergence-gpu-python convergence-gpu-julia \
        clean verify verify-separation bench-overhead summary check-docs

# Benchmarks must run sequentially (1 language at a time) to avoid
# AMX/thermal contention that would invalidate measurements.
.NOTPARALLEL:

# --- Configuration ---
ROOT     := $(shell pwd)
JULIA_THREADS ?= 4
BENCH_COOL ?= 15
BENCH_TRIALS ?= 10
BENCH_WARMUP ?= 3

# Colors (only when terminal supports it)
OK  := \033[32m✓\033[0m
ERR := \033[31m✗\033[0m
HDR := \033[1;36m

# --- Test targets ---
test: test-rust test-go test-python test-julia
	@printf "\n$(HDR)All tests passed.$(OK)\033[0m\n"

test-rust:
	@printf "$(HDR)[Rust]   Testing...\033[0m "
	@cd $(ROOT)/rust && cargo test --quiet 2>&1 && printf "$(OK)\n" || (printf "$(ERR)\n" && exit 1)

test-go:
	@printf "$(HDR)[Go]     Testing...\033[0m "
	@cd $(ROOT)/go && go test -run 'Test[^B]' -count=1 ./... > /dev/null 2>&1 && printf "$(OK)\n" || (printf "$(ERR)\n" && exit 1)

test-python:
	@printf "$(HDR)[Python] Testing...\033[0m "
	@cd $(ROOT)/python && python3 -m pytest tests/ -q --tb=short 2>&1 | tail -1 && printf "$(OK)\n" || (printf "$(ERR)\n" && exit 1)

test-julia:
	@printf "$(HDR)[Julia]  Testing...\033[0m "
	@cd $(ROOT)/julia && julia --project=. test/runtests.jl > /dev/null 2>&1 && printf "$(OK)\n" || (printf "$(ERR)\n" && exit 1)

# --- CPU Benchmark targets ---
bench-cpu: bench-cpu-rust bench-cool bench-cpu-go bench-cool bench-cpu-python bench-cool bench-cpu-julia summary
	@printf "\n$(HDR)All CPU benchmarks complete. Results in benchmarks/*.json$(OK)\033[0m\n"

bench-cool:
	@printf "  Cooling $(BENCH_COOL)s...\n"
	@sleep $(BENCH_COOL)

bench-cpu-rust:
	@printf "$(HDR)[Rust]   CPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/rust && ROSETTA_CPU_ONLY=1 ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) cargo run --release --bin bench 2>/dev/null > $(ROOT)/benchmarks/rust.json
	@printf "  $(OK) benchmarks/rust.json\n"

bench-cpu-go:
	@printf "$(HDR)[Go]     CPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/go && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) go test -run=TestBenchCPU -v -count=1 -timeout=600s ./cpu 2>/dev/null | awk '/^\{$$/,/^\}$$/' > $(ROOT)/benchmarks/go.json
	@printf "  $(OK) benchmarks/go.json\n"

bench-cpu-python:
	@printf "$(HDR)[Python] CPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/python && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) python3 bench_cpu.py > $(ROOT)/benchmarks/python.json 2>/dev/null
	@printf "  $(OK) benchmarks/python.json\n"

bench-cpu-julia:
	@printf "$(HDR)[Julia]  CPU Benchmarking...\033[0m\n"
	@cd $(ROOT) && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) JULIA_NUM_THREADS=$(JULIA_THREADS) julia --project=julia julia/bench_cpu.jl > $(ROOT)/benchmarks/julia.json 2>/dev/null
	@printf "  $(OK) benchmarks/julia.json\n"

# --- Summary: extract key numbers from JSON ---
summary:
	@printf "\n$(HDR)=== Benchmark Summary ===$(OK)\033[0m\n"
	@python3 $(ROOT)/scripts/summary.py

# --- Convergence verification ---
convergence: convergence-rust convergence-go convergence-python convergence-julia
	@printf "\n$(HDR)=== Convergence Results ===$(OK)\033[0m\n"

convergence-rust:
	@printf "$(HDR)[Rust]   Convergence...\033[0m "
	@cd $(ROOT)/rust && cargo run --release --bin convergence 2>/dev/null | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-go:
	@printf "$(HDR)[Go]     Convergence...\033[0m "
	@cd $(ROOT)/go && go test -run TestConvergence -v 2>&1 | grep -o '"losses":\[.*\]' | python3 -c "import sys,json;s=sys.stdin.read();l=json.loads('{'+s+'}')['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-python:
	@printf "$(HDR)[Python] Convergence...\033[0m "
	@cd $(ROOT) && python3 scripts/convergence_python.py | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-julia:
	@printf "$(HDR)[Julia]  Convergence...\033[0m "
	@cd $(ROOT) && julia scripts/convergence_julia.jl | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

# --- GPU convergence verification ---
convergence-gpu: convergence-gpu-rust convergence-gpu-go convergence-gpu-python convergence-gpu-julia
	@printf "\n$(HDR)=== GPU Convergence Results ===$(OK)\033[0m\n"

convergence-gpu-rust:
	@printf "$(HDR)[Rust]   GPU Convergence...\033[0m "
	@cd $(ROOT)/rust && cargo run --release --features metal --bin convergence_gpu 2>/dev/null | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-gpu-go:
	@printf "$(HDR)[Go]     GPU Convergence...\033[0m "
	@cd $(ROOT)/go && go test -run TestConvergenceGpu -v 2>&1 | grep -o '"losses":\[.*\]' | python3 -c "import sys,json;s=sys.stdin.read();l=json.loads('{'+s+'}')['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-gpu-python:
	@printf "$(HDR)[Python] GPU Convergence...\033[0m "
	@cd $(ROOT) && python3 scripts/convergence_python_gpu.py | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-gpu-julia:
	@printf "$(HDR)[Julia]  GPU Convergence...\033[0m "
	@cd $(ROOT) && julia --project=julia scripts/convergence_julia_gpu.jl | python3 -c "import sys,json;d=json.loads(sys.stdin.read());l=d['losses'];print(f'{l[0]:.4f} -> {l[-1]:.4f} (delta={l[-1]-l[0]:+.4f})')"

convergence-plots:
	@printf "$(HDR)[All]    Convergence + per-language SVG plots...\033[0m\n"
	@cd $(ROOT) && python3 scripts/convergence_plots.py

# --- Doc verification: JSON vs docs ---
check-docs:
	@printf "\n$(HDR)=== Checking docs vs JSON ===$(OK)\033[0m\n"
	@python3 $(ROOT)/scripts/check_docs.py

# --- Full verification ---
verify: verify-separation test bench-cpu convergence check-docs
	@printf "\n$(HDR)=== Verification Complete ===$(OK)\033[0m\n"

# --- GPU benchmark targets (requires Metal) ---
bench-gpu: bench-gpu-rust bench-cool bench-gpu-go bench-cool bench-gpu-python bench-cool bench-gpu-julia summary
	@printf "\n$(HDR)All GPU benchmarks complete.$(OK)\033[0m\n"

bench-gpu-rust:
	@printf "$(HDR)[Rust]   GPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/rust && ROSETTA_GPU_ONLY=1 ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) cargo run --release --features metal --bin bench 2>/dev/null > $(ROOT)/benchmarks/rust_gpu.json
	@printf "  $(OK) benchmarks/rust_gpu.json\n"

bench-gpu-julia:
	@printf "$(HDR)[Julia]  GPU Benchmarking...\033[0m\n"
	@cd $(ROOT) && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) JULIA_NUM_THREADS=$(JULIA_THREADS) julia --project=julia julia/bench_gpu.jl > $(ROOT)/benchmarks/julia_gpu.json 2>/dev/null
	@printf "  $(OK) benchmarks/julia_gpu.json\n"

bench-gpu-go:
	@printf "$(HDR)[Go]     GPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/go && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) go test -run=TestBenchGPU -v -count=1 -timeout=600s ./gpu 2>/dev/null | awk '/^\{$$/,/^\}$$/' > $(ROOT)/benchmarks/go_gpu.json
	@printf "  $(OK) benchmarks/go_gpu.json\n"

bench-gpu-python:
	@printf "$(HDR)[Python] GPU Benchmarking...\033[0m\n"
	@cd $(ROOT)/python && ROSETTA_BENCH_TRIALS=$(BENCH_TRIALS) ROSETTA_BENCH_WARMUP=$(BENCH_WARMUP) python3 bench_gpu.py > $(ROOT)/benchmarks/python_gpu.json 2>/dev/null
	@printf "  $(OK) benchmarks/python_gpu.json\n"

# --- Full benchmark sweep: CPU + GPU sequential (8 runs) ---
bench-all:
	@$(MAKE) bench-cpu-rust
	@$(MAKE) bench-cool
	@$(MAKE) bench-cpu-go
	@$(MAKE) bench-cool
	@$(MAKE) bench-cpu-python
	@$(MAKE) bench-cool
	@$(MAKE) bench-cpu-julia
	@$(MAKE) bench-cool
	@$(MAKE) bench-gpu-rust
	@$(MAKE) bench-cool
	@$(MAKE) bench-gpu-go
	@$(MAKE) bench-cool
	@$(MAKE) bench-gpu-python
	@$(MAKE) bench-cool
	@$(MAKE) bench-gpu-julia
	@$(MAKE) summary
	@printf "\n$(HDR)All CPU+GPU benchmarks complete (8 runs).$(OK)\033[0m\n"

bench-all-30:
	@printf "$(HDR)Running full benchmark sweep with fixed 30 trials...\033[0m\n"
	@$(MAKE) bench-all BENCH_TRIALS=30

# --- CPU/GPU separation verification ---
verify-separation:
	@printf "\n$(HDR)=== CPU/GPU Separation Check ===$(OK)\033[0m\n"
	@$(ROOT)/scripts/check_separation.sh

# --- Language overhead analysis ---
bench-overhead:
	@printf "\n$(HDR)=== Language Overhead Analysis ===$(OK)\033[0m\n"
	@python3 $(ROOT)/scripts/measure_overhead.py

# --- Clean ---
clean:
	@printf "Cleaning build artifacts...\n"
	@rm -rf $(ROOT)/rust/target
	@rm -rf $(ROOT)/python/__pycache__ $(ROOT)/python/tests/__pycache__
	@rm -rf $(ROOT)/python/*.egg-info $(ROOT)/python/dist $(ROOT)/python/build
	@find $(ROOT) -name "*.pyc" -delete 2>/dev/null || true
	@find $(ROOT) -name "*.log" -delete 2>/dev/null || true
	@find $(ROOT) -name ".DS_Store" -delete 2>/dev/null || true
	@printf "$(OK) Clean complete.\n"
