#!/bin/bash
# Cross-language benchmark runner
# Runs benchmarks for Rust, Go, and Python

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  Cross-Language Benchmark Suite"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Rust benchmarks
run_rust() {
    echo -e "${BLUE}[Rust]${NC} Running criterion benchmarks..."
    cd "$SCRIPT_DIR/rust"

    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}[Rust]${NC} cargo not found, skipping"
        return 1
    fi

    cargo bench --quiet 2>&1 | tee "$RESULTS_DIR/rust_output.txt"

    # Extract results to JSON (simplified)
    echo -e "${GREEN}[Rust]${NC} Benchmarks complete"
}

# Go benchmarks
run_go() {
    echo -e "${BLUE}[Go]${NC} Running testing.B benchmarks..."
    cd "$SCRIPT_DIR/go"

    if ! command -v go &> /dev/null; then
        echo -e "${RED}[Go]${NC} go not found, skipping"
        return 1
    fi

    go test -bench=. -benchmem -count=5 2>&1 | tee "$RESULTS_DIR/go_output.txt"

    echo -e "${GREEN}[Go]${NC} Benchmarks complete"
}

# Python benchmarks
run_python() {
    echo -e "${BLUE}[Python]${NC} Running timeit benchmarks..."
    cd "$SCRIPT_DIR/python"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}[Python]${NC} python3 not found, skipping"
        return 1
    fi

    python3 bench_ops.py 2>&1 | tee "$RESULTS_DIR/python_output.txt"
    cp results_python.json "$RESULTS_DIR/" 2>/dev/null || true

    echo -e "${GREEN}[Python]${NC} Benchmarks complete"
}

# Parse arguments
RUN_RUST=true
RUN_GO=true
RUN_PYTHON=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --rust-only)
            RUN_GO=false
            RUN_PYTHON=false
            shift
            ;;
        --go-only)
            RUN_RUST=false
            RUN_PYTHON=false
            shift
            ;;
        --python-only)
            RUN_RUST=false
            RUN_GO=false
            shift
            ;;
        --no-rust)
            RUN_RUST=false
            shift
            ;;
        --no-go)
            RUN_GO=false
            shift
            ;;
        --no-python)
            RUN_PYTHON=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --rust-only    Run only Rust benchmarks"
            echo "  --go-only      Run only Go benchmarks"
            echo "  --python-only  Run only Python benchmarks"
            echo "  --no-rust      Skip Rust benchmarks"
            echo "  --no-go        Skip Go benchmarks"
            echo "  --no-python    Skip Python benchmarks"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run benchmarks
echo "Starting benchmarks at $(date)"
echo ""

if $RUN_RUST; then
    run_rust || echo -e "${RED}[Rust]${NC} Failed"
    echo ""
fi

if $RUN_GO; then
    run_go || echo -e "${RED}[Go]${NC} Failed"
    echo ""
fi

if $RUN_PYTHON; then
    run_python || echo -e "${RED}[Python]${NC} Failed"
    echo ""
fi

echo "=============================================="
echo "  All benchmarks complete!"
echo "  Results saved to: $RESULTS_DIR/"
echo "=============================================="
echo ""
echo "Files:"
ls -la "$RESULTS_DIR/"
