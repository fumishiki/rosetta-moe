# Cross-Language Benchmark Suite

Rust / Go / Python の純粋計算オーバーヘッドを比較するベンチマークスイート。

## Structure

```
benchmarks/
├── run_all.sh          # 全言語実行スクリプト
├── results/            # 出力結果
├── rust/               # Rust (criterion)
│   ├── Cargo.toml
│   ├── src/lib.rs
│   └── benches/tensor_ops.rs
├── go/                 # Go (testing.B)
│   ├── go.mod
│   ├── ops.go
│   └── ops_test.go
└── python/             # Python (timeit + numpy)
    └── bench_ops.py
```

## Benchmarks

| ベンチマーク | 内容 | サイズ |
|-------------|------|--------|
| `matmul` | 行列積 (naive) | 64, 128, 256, 512 |
| `softmax` | Row-wise softmax | 64x1024 ~ 512x32000 |
| `silu` | SiLU activation | 1K ~ 64K |
| `rmsnorm` | RMSNorm | 64x768 ~ 512x768 |

## Usage

### Run All

```bash
./run_all.sh
```

### Run Specific Language

```bash
./run_all.sh --rust-only
./run_all.sh --go-only
./run_all.sh --python-only
```

### Skip Language

```bash
./run_all.sh --no-python
```

### Individual Runs

```bash
# Rust
cd rust && cargo bench

# Go
cd go && go test -bench=. -benchmem

# Python
cd python && python3 bench_ops.py
```

## Expected Results

**Note:** Python uses NumPy with BLAS backend (OpenBLAS/MKL), which is highly optimized.

```
┌─────────────────┬──────────┬──────────┬──────────┐
│ Benchmark       │ Rust     │ Go       │ Python   │
├─────────────────┼──────────┼──────────┼──────────┤
│ matmul_256      │ ~5ms     │ ~8ms     │ ~0.2ms*  │
│ softmax_256x1K  │ ~0.5ms   │ ~0.8ms   │ ~0.1ms*  │
│ silu_16K        │ ~20µs    │ ~30µs    │ ~10µs*   │
│ rmsnorm_256x768 │ ~0.3ms   │ ~0.5ms   │ ~0.05ms* │
└─────────────────┴──────────┴──────────┴──────────┘
* NumPy uses BLAS (vectorized C/Fortran)
```

## Interpretation

| 言語 | 特徴 |
|------|------|
| **Rust** | ゼロコスト抽象化、予測可能なパフォーマンス |
| **Go** | GC pause、シンプルだがRust比やや遅い |
| **Python** | インタプリタオーバーヘッド大、NumPy経由でBLAS使用時は高速 |

### Key Insights

1. **純粋ループ計算** → Rust > Go >> Python
2. **BLAS使用時** → Python (NumPy) が最速（C/Fortran実装）
3. **GCの影響** → Go は大量アロケーション時にpause発生
4. **CUDA使用時** → 言語差は無視できる（カーネル実行時間が支配的）
