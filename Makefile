# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

# machine_learning — 4-language MoE Transformer benchmark
# Usage:
#   make test          Run all tests (Rust + Go + Python + Julia)
#   make bench         Run all benchmarks and save JSON
#   make convergence   Run loss convergence verification (all 4 languages)
#   make test-rust     Run Rust tests only
#   make bench-julia   Run Julia benchmark only
#   make clean         Remove build artifacts and caches
#   make verify        Full verification: test + bench + convergence + summary

.PHONY: test test-rust test-go test-python test-julia \
        bench bench-rust bench-go bench-python bench-julia bench-cool \
        convergence convergence-rust convergence-go convergence-python convergence-julia \
        clean verify summary check-docs

# --- Configuration ---
ROOT     := $(shell pwd)
JULIA_THREADS ?= 4
BENCH_COOL ?= 15

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

# --- Benchmark targets ---
bench: bench-rust bench-cool bench-go bench-cool bench-python bench-cool bench-julia summary
	@printf "\n$(HDR)All benchmarks complete. Results in benchmarks/*.json$(OK)\033[0m\n"

bench-cool:
	@printf "  Cooling $(BENCH_COOL)s...\n"
	@sleep $(BENCH_COOL)

bench-rust:
	@printf "$(HDR)[Rust]   Benchmarking...\033[0m\n"
	@cd $(ROOT)/rust && cargo run --release --bin bench 2>/dev/null > $(ROOT)/benchmarks/rust.json
	@printf "  $(OK) benchmarks/rust.json\n"

bench-go:
	@printf "$(HDR)[Go]     Benchmarking...\033[0m\n"
	@cd $(ROOT)/go && go test -run=TestBench -v -count=1 -timeout=600s ./... 2>/dev/null | awk '/^\{$$/,/^\}$$/' > $(ROOT)/benchmarks/go.json
	@printf "  $(OK) benchmarks/go.json\n"

bench-python:
	@printf "$(HDR)[Python] Benchmarking...\033[0m\n"
	@cd $(ROOT)/python && python3 bench.py > $(ROOT)/benchmarks/python.json 2>/dev/null
	@printf "  $(OK) benchmarks/python.json\n"

bench-julia:
	@printf "$(HDR)[Julia]  Benchmarking...\033[0m\n"
	@cd $(ROOT) && JULIA_NUM_THREADS=$(JULIA_THREADS) julia --project=julia julia/bench.jl > $(ROOT)/benchmarks/julia.json 2>/dev/null
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

# --- Doc verification: JSON vs docs ---
check-docs:
	@printf "\n$(HDR)=== Checking docs vs JSON ===$(OK)\033[0m\n"
	@python3 $(ROOT)/scripts/check_docs.py

# --- Full verification ---
verify: test bench convergence check-docs
	@printf "\n$(HDR)=== Verification Complete ===$(OK)\033[0m\n"

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
