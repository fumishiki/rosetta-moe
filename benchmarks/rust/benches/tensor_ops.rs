use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use bench_rust::{matmul, softmax, silu, rmsnorm, random_vec};

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [64, 128, 256, 512] {
        let a = random_vec(size * size, 42);
        let b = random_vec(size * size, 123);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bencher, &size| {
                bencher.iter(|| {
                    black_box(matmul(&a, &b, size, size, size))
                });
            },
        );
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    for (rows, cols) in [(64, 1024), (128, 1024), (256, 1024), (512, 32000)] {
        let input = random_vec(rows * cols, 42);

        group.bench_with_input(
            BenchmarkId::new("row_wise", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |bencher, &(rows, cols)| {
                bencher.iter(|| {
                    black_box(softmax(&input, rows, cols))
                });
            },
        );
    }
    group.finish();
}

fn bench_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu");

    for size in [1024, 4096, 16384, 65536] {
        let input = random_vec(size, 42);

        group.bench_with_input(
            BenchmarkId::new("elementwise", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(silu(&input))
                });
            },
        );
    }
    group.finish();
}

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    for (batch_seq, dim) in [(64, 768), (128, 768), (256, 768), (512, 768)] {
        let input = random_vec(batch_seq * dim, 42);
        let weight = random_vec(dim, 123);

        group.bench_with_input(
            BenchmarkId::new("norm", format!("{}x{}", batch_seq, dim)),
            &(batch_seq, dim),
            |bencher, &(_, dim)| {
                bencher.iter(|| {
                    black_box(rmsnorm(&input, &weight, dim, 1e-6))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_softmax, bench_silu, bench_rmsnorm);
criterion_main!(benches);
