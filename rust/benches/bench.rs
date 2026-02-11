// SPDX-License-Identifier: CC-BY-NC-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Benchmark harness for the MoE Transformer.
//!
//! Custom harness (not Criterion) to produce a standardized JSON output
//! that is comparable across all 4 language implementations.
//!
//! 5 benchmark axes, 22 total scenarios:
//! 1. **Memory Management** (9): train step, batch scaling [1,2,4,8], seq scaling [8,16,32,64]
//! 2. **Compiler Optimization** (3): matmul, softmax, rmsnorm kernels
//! 3. **Type System / Dispatch** (2): warm (reuse model), cold (fresh model each trial)
//! 4. **Parallelism** (6): 1, 2, 4 threads running independent forward passes + train steps
//! 5. **Scale Comparison** (2): forward and train at hidden=256 vs tiny hidden=64
//!
//! Each scenario runs N_WARMUP warmup iterations + N_TRIALS timed iterations,
//! collecting wall-clock time, CPU time (user+sys), and allocation bytes.
//!
//! Allocation tracking: a custom global allocator (`CountingAlloc`) wraps the
//! system allocator, using a thread-local counter to track allocation bytes.
//! This avoids cache-line bouncing under parallel allocation pressure.
//! Measures heap bytes allocated per trial with zero overhead on dealloc.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use nn_core::{Config, DType, Layer, MoETransformer, RMSNorm, Shape, Tensor, TrainConfig, Trainer};

thread_local! {
    static TL_ALLOC_BYTES: Cell<u64> = const { Cell::new(0) };
}

/// Global allocator wrapper that counts total bytes allocated.
struct CountingAlloc;

// SAFETY: Delegates all allocation/deallocation to System allocator.
// Thread-local counter avoids cache-line bouncing under parallel allocation
// pressure. `try_with` skips counting if TLS is not yet initialized (early
// startup / thread teardown).
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TL_ALLOC_BYTES.try_with(|c| c.set(c.get() + layout.size() as u64));
        // SAFETY: Forwarding to System allocator with the same layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr was allocated by System.alloc with the same layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

const SEED: u64 = 42;
const VOCAB: usize = 1000;
const HIDDEN: usize = 64;
const N_TRIALS: usize = 10;
const N_WARMUP: usize = 3;

fn make_token_ids(batch: usize, seq_len: usize, vocab: usize) -> Vec<usize> {
    (0..batch * seq_len).map(|i| i % vocab).collect()
}

fn make_input_tensor(batch: usize, seq_len: usize, vocab: usize) -> Tensor {
    let data: Vec<f32> = (0..batch * seq_len).map(|i| (i % vocab) as f32).collect();
    Tensor::from_slice(&data, Shape::new(&[batch, seq_len]))
}

fn make_targets_tensor(batch: usize, seq_len: usize, vocab: usize) -> Tensor {
    let data: Vec<f32> = (0..batch * seq_len)
        .map(|i| ((i + 1) % vocab) as f32)
        .collect();
    Tensor::from_slice(&data, Shape::new(&[batch, seq_len]))
}

fn reset_alloc_counter() {
    TL_ALLOC_BYTES.with(|c| c.set(0));
}

fn read_alloc_bytes() -> u64 {
    TL_ALLOC_BYTES.with(|c| c.get())
}

/// Get cumulative CPU time (user + system) via getrusage.
/// Used to measure actual CPU work vs wall-clock time (which includes I/O waits).
fn get_cpu_time_ns() -> u64 {
    // SAFETY: rusage is a plain-old-data struct; zeroing all bytes is a valid initial state.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage(RUSAGE_SELF, &mut usage) is always safe -- it writes
    // process-level resource usage into a stack-allocated struct we own.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return 0;
    }
    let user_ns =
        usage.ru_utime.tv_sec as u64 * 1_000_000_000 + usage.ru_utime.tv_usec as u64 * 1000;
    let sys_ns =
        usage.ru_stime.tv_sec as u64 * 1_000_000_000 + usage.ru_stime.tv_usec as u64 * 1000;
    user_ns + sys_ns
}

/// Peak resident set size (RSS) in bytes.
/// Note: macOS reports ru_maxrss in bytes, Linux in kilobytes.
fn get_peak_rss_bytes() -> u64 {
    // SAFETY: rusage is a plain-old-data struct; zeroing all bytes is a valid initial state.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: Same as get_cpu_time_ns -- safe getrusage call.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret == 0 {
        if cfg!(target_os = "macos") {
            usage.ru_maxrss as u64 // macOS: already in bytes
        } else {
            usage.ru_maxrss as u64 * 1024 // Linux: in kilobytes
        }
    } else {
        0
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let k = (n - 1) as f64 * p / 100.0;
    let f = k.floor() as usize;
    let c = f + 1;
    if c >= n {
        return sorted[n - 1];
    }
    let lower = sorted[f] as f64;
    let upper = sorted[c] as f64;
    (lower + (k - f as f64) * (upper - lower)).round() as u64
}

fn stats(timings: &[u64]) -> (u64, u64, u64, u64, u64) {
    let mut sorted: Vec<u64> = timings.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let min = sorted[0];
    let max = sorted[n - 1];
    let median = if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    } else {
        sorted[n / 2]
    };
    let p95 = percentile(&sorted, 95.0);
    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;
    (median, p95, min, max, iqr)
}

fn check_numerical(t: &Tensor) -> (usize, usize, f32) {
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;
    let mut max_abs = 0.0f32;
    for &v in t.data() {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        }
        let a = v.abs();
        if a.is_finite() && a > max_abs {
            max_abs = a;
        }
    }
    (nan_count, inf_count, max_abs)
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

fn timings_json(timings: &[u64]) -> String {
    let parts: Vec<String> = timings.iter().map(|t| t.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn get_cpu_model() -> String {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();
    match output {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

fn get_os_info() -> String {
    let output = std::process::Command::new("uname").args(["-srm"]).output();
    match output {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s
            }
        }
        Err(_) => "unknown".to_string(),
    }
}

fn get_timestamp() -> String {
    let output = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output();
    match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".to_string(),
    }
}

// ─── Measurement ────────────────────────────────────────────────────────────

struct Measurement {
    timings_ns: Vec<u64>,
    cpu_times_ns: Vec<u64>,
    warmup_timings_ns: Vec<u64>,
    alloc_bytes: u64,
}

fn measure(n_warmup: usize, n_trials: usize, mut f: impl FnMut()) -> Measurement {
    let mut warmup_timings_ns = Vec::with_capacity(n_warmup);
    for _ in 0..n_warmup {
        let start = Instant::now();
        f();
        warmup_timings_ns.push(start.elapsed().as_nanos() as u64);
    }

    let mut timings_ns = Vec::with_capacity(n_trials);
    let mut cpu_times_ns = Vec::with_capacity(n_trials);
    let mut total_alloc = 0u64;

    for _ in 0..n_trials {
        reset_alloc_counter();
        let cpu_before = get_cpu_time_ns();
        let start = Instant::now();
        f();
        let wall_ns = start.elapsed().as_nanos() as u64;
        let cpu_after = get_cpu_time_ns();
        timings_ns.push(wall_ns);
        cpu_times_ns.push(cpu_after.saturating_sub(cpu_before));
        total_alloc += read_alloc_bytes();
    }

    let avg_alloc = if n_trials > 0 {
        total_alloc / n_trials as u64
    } else {
        0
    };

    Measurement {
        timings_ns,
        cpu_times_ns,
        warmup_timings_ns,
        alloc_bytes: avg_alloc,
    }
}

// ─── Scenario JSON builder ──────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn scenario_json(
    id: &str,
    axis: &str,
    params: &str,
    m: &Measurement,
    nan_count: usize,
    inf_count: usize,
    max_abs: Option<f32>,
    known_flops: Option<u64>,
    batch_seq_tokens: Option<usize>,
    n_warmup_override: Option<usize>,
) -> String {
    let warmup_runs = n_warmup_override.unwrap_or(N_WARMUP);
    let (median_ns, p95_ns, min_ns, max_ns, iqr_ns) = stats(&m.timings_ns);

    let cpu_median_ns = {
        let mut sorted = m.cpu_times_ns.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        if n == 0 {
            0
        } else if n.is_multiple_of(2) {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        } else {
            sorted[n / 2]
        }
    };

    let peak_rss = get_peak_rss_bytes();

    let throughput_tokens_per_sec = match batch_seq_tokens {
        Some(tok) if median_ns > 0 => {
            format!("{:.2}", tok as f64 / (median_ns as f64 * 1e-9))
        }
        _ => "null".to_string(),
    };

    let alloc_rate = if median_ns > 0 {
        format!("{:.2}", m.alloc_bytes as f64 / (median_ns as f64 * 1e-9))
    } else {
        "0.0".to_string()
    };

    let gflops_str = match known_flops {
        Some(flops) if median_ns > 0 => {
            let gf = flops as f64 / (median_ns as f64 * 1e-9) * 1e-9;
            format!("{gf:.4}")
        }
        _ => "null".to_string(),
    };

    let max_abs_str = match max_abs {
        Some(v) if v.is_finite() => format!("{v:.6}"),
        _ => "null".to_string(),
    };

    format!(
        concat!(
            "{{",
            "\"id\":\"{id}\",",
            "\"axis\":\"{axis}\",",
            "\"params\":{params},",
            "\"warmup_runs\":{warmup_runs},",
            "\"trial_runs\":{trial_runs},",
            "\"timings_ns\":{timings},",
            "\"cpu_times_ns\":{cpu_times},",
            "\"warmup_timings_ns\":{warmup_timings},",
            "\"median_ns\":{median},",
            "\"p95_ns\":{p95},",
            "\"min_ns\":{min},",
            "\"max_ns\":{max},",
            "\"iqr_ns\":{iqr},",
            "\"cpu_median_ns\":{cpu_median},",
            "\"memory\":{{\"peak_rss_bytes\":{rss},\"alloc_bytes\":{alloc}}},",
            "\"gc\":{{\"total_gc_time_ns\":0,\"gc_pause_count\":0}},",
            "\"numerical\":{{\"nan_count\":{nan},\"inf_count\":{inf},\"max_abs\":{max_abs}}},",
            "\"throughput_tokens_per_sec\":{throughput},",
            "\"derived\":{{\"alloc_rate_bytes_per_sec\":{alloc_rate},\"gc_throughput\":1.0{gflops_field}}}",
            "}}"
        ),
        id = id,
        axis = axis,
        params = params,
        warmup_runs = warmup_runs,
        trial_runs = m.timings_ns.len(),
        timings = timings_json(&m.timings_ns),
        cpu_times = timings_json(&m.cpu_times_ns),
        warmup_timings = timings_json(&m.warmup_timings_ns),
        median = median_ns,
        p95 = p95_ns,
        min = min_ns,
        max = max_ns,
        iqr = iqr_ns,
        cpu_median = cpu_median_ns,
        rss = peak_rss,
        alloc = m.alloc_bytes,
        nan = nan_count,
        inf = inf_count,
        max_abs = max_abs_str,
        throughput = throughput_tokens_per_sec,
        alloc_rate = alloc_rate,
        gflops_field = if gflops_str != "null" {
            format!(",\"gflops\":{gflops_str}")
        } else {
            String::new()
        },
    )
}

// ─── Axis 1: Memory Management ─────────────────────────────────────────────

fn bench_mem_train_step() -> String {
    eprintln!("  [memory] mem_train_step");
    let batch = 2;
    let seq = 8;
    let model = MoETransformer::tiny();
    let train_cfg = TrainConfig {
        batch_size: batch,
        seq_len: seq,
        lr: 1e-4,
        warmup_steps: 0,
        total_steps: 100,
        grad_clip: 1.0,
        aux_loss_weight: 0.01,
    };
    let mut trainer = Trainer::new(model, train_cfg);
    let input = make_input_tensor(batch, seq, VOCAB);
    let targets = make_targets_tensor(batch, seq, VOCAB);

    let m = measure(N_WARMUP, N_TRIALS, || {
        let _ = trainer.train_step(&input, &targets);
    });

    let params = format!("{{\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN}}}");
    scenario_json(
        "mem_train_step",
        "memory",
        &params,
        &m,
        0,
        0,
        None,
        None,
        Some(batch * seq),
        None,
    )
}

fn bench_mem_scale_batch() -> Vec<String> {
    let batches = [1, 2, 4, 8];
    let seq = 32;
    let mut results = Vec::new();

    for &b in &batches {
        eprintln!("  [memory] mem_scale_batch_{b}");
        let mut model = MoETransformer::tiny();
        model.set_inference_mode(true);
        let ids = make_token_ids(b, seq, VOCAB);
        let mut last_output = None;

        let m = measure(N_WARMUP, N_TRIALS, || {
            last_output = Some(model.forward_ids(&ids, b, seq));
        });

        let (nan, inf, ma) = match &last_output {
            Some(t) => check_numerical(t),
            None => (0, 0, 0.0),
        };

        let params = format!("{{\"batch\":{b},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN}}}");
        results.push(scenario_json(
            &format!("mem_scale_batch_{b}"),
            "memory",
            &params,
            &m,
            nan,
            inf,
            Some(ma),
            None,
            Some(b * seq),
            None,
        ));
    }
    results
}

fn bench_mem_scale_seq() -> Vec<String> {
    let seqs = [8, 16, 32, 64];
    let batch = 2;
    let mut results = Vec::new();

    for &s in &seqs {
        eprintln!("  [memory] mem_scale_seq_{s}");
        let mut model = MoETransformer::tiny();
        model.set_inference_mode(true);
        let ids = make_token_ids(batch, s, VOCAB);
        let mut last_output = None;

        let m = measure(N_WARMUP, N_TRIALS, || {
            last_output = Some(model.forward_ids(&ids, batch, s));
        });

        let (nan, inf, ma) = match &last_output {
            Some(t) => check_numerical(t),
            None => (0, 0, 0.0),
        };

        let params = format!("{{\"batch\":{batch},\"seq_len\":{s},\"hidden_dim\":{HIDDEN}}}");
        results.push(scenario_json(
            &format!("mem_scale_seq_{s}"),
            "memory",
            &params,
            &m,
            nan,
            inf,
            Some(ma),
            None,
            Some(batch * s),
            None,
        ));
    }
    results
}

// ─── Axis 2: Compiler Optimization (kernel benchmarks) ──────────────────────
// These benchmarks isolate individual kernels (matmul, softmax, rmsnorm)
// to measure raw compute performance without model-level overhead.

fn bench_kernel_matmul() -> String {
    eprintln!("  [compiler] kernel_matmul");
    let m_dim = 64;
    let k_dim = 64;
    let n_dim = 64;
    // Matmul: C = A @ B, where A:[M,K], B:[K,N], C:[M,N]
    // FLOPs = 2*M*N*K = 2*64*64*64 = 524288 (multiply-add = 2 FLOPs)
    let a = Tensor::randn(Shape::new(&[m_dim, k_dim]), DType::F32, SEED);
    let b = Tensor::randn(Shape::new(&[k_dim, n_dim]), DType::F32, SEED + 1);
    // Pre-allocate output buffer OUTSIDE the timing loop to measure pure BLAS time.
    // Rust's try_matmul has overhead (batch_dims comparison, shape construction)
    // so we call sgemm directly for a fair kernel-level comparison with Go/Julia.
    let mut out = vec![0.0f32; m_dim * n_dim];

    let meas = measure(N_WARMUP, N_TRIALS, || {
        nn_core::sgemm(m_dim, n_dim, k_dim, 1.0, a.data(), b.data(), 0.0, &mut out);
    });

    let out_tensor = Tensor::from_slice(&out, Shape::new(&[m_dim, n_dim]));
    let (nan, inf, ma) = check_numerical(&out_tensor);

    // known_flops = 2 * M * N * K = 2 * 64 * 64 * 64 = 524288
    let params = format!("{{\"M\":{m_dim},\"K\":{k_dim},\"N\":{n_dim}}}");
    scenario_json(
        "kernel_matmul",
        "compiler",
        &params,
        &meas,
        nan,
        inf,
        Some(ma),
        Some(524288),
        None,
        None,
    )
}

fn bench_kernel_softmax() -> String {
    eprintln!("  [compiler] kernel_softmax");
    let n = 1000;
    let input = Tensor::randn(Shape::new(&[1, n]), DType::F32, SEED);
    let mut buf = vec![0.0f32; n];

    let meas = measure(N_WARMUP, N_TRIALS, || {
        buf.copy_from_slice(input.data());
        nn_core::softmax_in_place(&mut buf);
    });

    let out_tensor = Tensor::from_slice(&buf, Shape::new(&[1, n]));
    let (nan, inf, ma) = check_numerical(&out_tensor);

    // known_flops = 4 * n = 4000 (max + exp + sum + div per element)
    let params = format!("{{\"n\":{n}}}");
    scenario_json(
        "kernel_softmax",
        "compiler",
        &params,
        &meas,
        nan,
        inf,
        Some(ma),
        Some(4000),
        None,
        None,
    )
}

fn bench_kernel_rmsnorm() -> String {
    eprintln!("  [compiler] kernel_rmsnorm");
    let shape = [2, 32, HIDDEN];
    let input = Tensor::randn(Shape::new(&shape), DType::F32, SEED);
    let mut norm = RMSNorm::new(HIDDEN);
    let mut last_output = None;

    let meas = measure(N_WARMUP, N_TRIALS, || {
        last_output = Some(norm.forward(&input));
    });

    let (nan, inf, ma) = match &last_output {
        Some(t) => check_numerical(t),
        None => (0, 0, 0.0),
    };

    // known_flops = batch * seq * hidden * 3 = 2 * 32 * 64 * 3 = 12288
    let params = format!("{{\"shape\":[{},{},{}]}}", shape[0], shape[1], shape[2]);
    scenario_json(
        "kernel_rmsnorm",
        "compiler",
        &params,
        &meas,
        nan,
        inf,
        Some(ma),
        Some(12288),
        None,
        None,
    )
}

// ─── Axis 3: Type System (dispatch benchmarks) ─────────────────────────────

fn bench_dispatch_warm() -> String {
    eprintln!("  [type_system] dispatch_warm");
    let batch = 2;
    let seq = 32;
    let mut model = MoETransformer::tiny();
    model.set_inference_mode(true);
    let ids = make_token_ids(batch, seq, VOCAB);
    let mut last_output = None;

    let m = measure(N_WARMUP, N_TRIALS, || {
        last_output = Some(model.forward_ids(&ids, batch, seq));
    });

    let (nan, inf, ma) = match &last_output {
        Some(t) => check_numerical(t),
        None => (0, 0, 0.0),
    };

    let params = format!("{{\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN}}}");
    scenario_json(
        "dispatch_warm",
        "type_system",
        &params,
        &m,
        nan,
        inf,
        Some(ma),
        None,
        Some(batch * seq),
        None,
    )
}

fn bench_dispatch_cold() -> String {
    eprintln!("  [type_system] dispatch_cold");
    let batch = 1;
    let seq = 8;
    let ids = make_token_ids(batch, seq, VOCAB);

    let mut last_output = None;

    let m = measure(N_WARMUP, N_TRIALS, || {
        let cfg = Config::tiny();
        let mut model = MoETransformer::new(cfg);
        model.set_inference_mode(true);
        last_output = Some(model.forward_ids(&ids, batch, seq));
    });

    let (nan, inf, ma) = match &last_output {
        Some(t) => check_numerical(t),
        None => (0, 0, 0.0),
    };

    let params = format!("{{\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN}}}");
    scenario_json(
        "dispatch_cold",
        "type_system",
        &params,
        &m,
        nan,
        inf,
        Some(ma),
        None,
        Some(batch * seq),
        None,
    )
}

// ─── Axis 4: Parallel ──────────────────────────────────────────────────────
// Measures scaling from 1 to N threads, each running an independent forward pass.
// This tests whether the BLAS library and memory allocator scale well under
// thread contention.

fn bench_parallel() -> Vec<String> {
    use std::sync::Barrier;

    let thread_counts = [1, 2, 4];
    let batch = 2;
    let seq = 32;
    let ids = make_token_ids(batch, seq, VOCAB);
    let mut results = Vec::new();
    let total_iters = N_WARMUP + N_TRIALS;

    for &t in &thread_counts {
        eprintln!("  [parallel] parallel_T{t}");

        // Pre-create one model per thread OUTSIDE the timing loops.
        // Wrapped in Mutex because MoETransformer contains RefCell (for MoE route
        // caching), which makes it !Sync. Each thread locks its own Mutex so
        // there is zero contention -- the Mutex is only for Send/Sync compliance.
        let models: Vec<Mutex<MoETransformer>> = (0..t)
            .map(|_| {
                let mut m = MoETransformer::tiny();
                m.set_inference_mode(true);
                Mutex::new(m)
            })
            .collect();

        // Per-thread alloc accumulators (written by workers, read by main)
        let thread_allocs: Vec<AtomicU64> = (0..t).map(|_| AtomicU64::new(0)).collect();

        let mut warmup_timings_ns = Vec::with_capacity(N_WARMUP);
        let mut timings_ns = Vec::with_capacity(N_TRIALS);
        let mut cpu_times_ns = Vec::with_capacity(N_TRIALS);
        let mut total_alloc = 0u64;

        // One scope for entire scenario -- threads spawned ONCE, reused via barriers
        let barrier_go = Barrier::new(t + 1); // t workers + main
        let barrier_done = Barrier::new(t + 1);

        std::thread::scope(|s| {
            // Spawn worker threads (once, reused across all warmup+trial iterations)
            for i in 0..t {
                let barrier_go = &barrier_go;
                let barrier_done = &barrier_done;
                let model = &models[i];
                let ids_ref = &ids;
                let alloc_slot = &thread_allocs[i];

                s.spawn(move || {
                    for _ in 0..total_iters {
                        barrier_go.wait(); // wait for "go"
                        reset_alloc_counter(); // reset THIS thread's counter
                        let mut m = model.lock().expect("bench: mutex poisoned");
                        let _ = m.forward_ids(ids_ref, batch, seq);
                        drop(m);
                        alloc_slot.store(read_alloc_bytes(), Ordering::Release);
                        barrier_done.wait(); // signal "done"
                    }
                });
            }

            // Main thread coordinates warmup + trial iterations
            for iter in 0..total_iters {
                reset_alloc_counter(); // reset main thread counter (unused, but clean)
                let cpu_before = get_cpu_time_ns();
                let start = Instant::now();
                barrier_go.wait(); // release all workers
                barrier_done.wait(); // wait for all workers to finish
                let wall_ns = start.elapsed().as_nanos() as u64;
                let cpu_after = get_cpu_time_ns();

                if iter < N_WARMUP {
                    warmup_timings_ns.push(wall_ns);
                } else {
                    timings_ns.push(wall_ns);
                    cpu_times_ns.push(cpu_after.saturating_sub(cpu_before));
                    let iter_alloc: u64 = thread_allocs
                        .iter()
                        .map(|a| a.load(Ordering::Acquire))
                        .sum();
                    total_alloc += iter_alloc;
                }
            }
        }); // threads joined here (once)

        let avg_alloc = total_alloc / N_TRIALS as u64;
        let m = Measurement {
            timings_ns,
            cpu_times_ns,
            warmup_timings_ns,
            alloc_bytes: avg_alloc,
        };

        let total_tokens = t * batch * seq;
        let params = format!(
            "{{\"thread_count\":{t},\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN}}}"
        );
        results.push(scenario_json(
            &format!("parallel_T{t}"),
            "parallel",
            &params,
            &m,
            0,
            0,
            None,
            None,
            Some(total_tokens),
            None,
        ));
    }
    results
}

fn bench_parallel_train() -> Vec<String> {
    use std::sync::Barrier;

    let thread_counts = [1, 2, 4];
    let batch = 2;
    let seq = 8;
    let mut results = Vec::new();
    let total_iters = N_WARMUP + N_TRIALS;

    for &t in &thread_counts {
        eprintln!("  [parallel] parallel_train_T{t}");

        // Per-thread: independent Trainer (wraps MoETransformer + AdamW).
        // Trainer contains RefCell (via MoETransformer) → !Sync, so wrap in Mutex.
        // Each thread locks its own Mutex — zero contention.
        let trainers: Vec<Mutex<Trainer>> = (0..t)
            .map(|_| {
                let model = MoETransformer::tiny();
                let train_cfg = TrainConfig {
                    batch_size: batch,
                    seq_len: seq,
                    lr: 1e-4,
                    warmup_steps: 0,
                    total_steps: 100,
                    grad_clip: 1.0,
                    aux_loss_weight: 0.01,
                };
                Mutex::new(Trainer::new(model, train_cfg))
            })
            .collect();

        // Per-thread input/target tensors (owned, no sharing)
        let inputs: Vec<Tensor> = (0..t)
            .map(|_| make_input_tensor(batch, seq, VOCAB))
            .collect();
        let targets: Vec<Tensor> = (0..t)
            .map(|_| make_targets_tensor(batch, seq, VOCAB))
            .collect();

        let thread_allocs: Vec<AtomicU64> = (0..t).map(|_| AtomicU64::new(0)).collect();

        let mut warmup_timings_ns = Vec::with_capacity(N_WARMUP);
        let mut timings_ns = Vec::with_capacity(N_TRIALS);
        let mut cpu_times_ns = Vec::with_capacity(N_TRIALS);
        let mut total_alloc = 0u64;

        let barrier_go = Barrier::new(t + 1);
        let barrier_done = Barrier::new(t + 1);

        std::thread::scope(|s| {
            for i in 0..t {
                let barrier_go = &barrier_go;
                let barrier_done = &barrier_done;
                let trainer = &trainers[i];
                let input = &inputs[i];
                let target = &targets[i];
                let alloc_slot = &thread_allocs[i];

                s.spawn(move || {
                    for _ in 0..total_iters {
                        barrier_go.wait();
                        reset_alloc_counter();
                        let mut tr = trainer.lock().expect("bench: mutex poisoned");
                        let _ = tr.train_step(input, target);
                        drop(tr);
                        alloc_slot.store(read_alloc_bytes(), Ordering::Release);
                        barrier_done.wait();
                    }
                });
            }

            for iter in 0..total_iters {
                reset_alloc_counter();
                let cpu_before = get_cpu_time_ns();
                let start = Instant::now();
                barrier_go.wait();
                barrier_done.wait();
                let wall_ns = start.elapsed().as_nanos() as u64;
                let cpu_after = get_cpu_time_ns();

                if iter < N_WARMUP {
                    warmup_timings_ns.push(wall_ns);
                } else {
                    timings_ns.push(wall_ns);
                    cpu_times_ns.push(cpu_after.saturating_sub(cpu_before));
                    let iter_alloc: u64 = thread_allocs
                        .iter()
                        .map(|a| a.load(Ordering::Acquire))
                        .sum();
                    total_alloc += iter_alloc;
                }
            }
        });

        let avg_alloc = total_alloc / N_TRIALS as u64;
        let m = Measurement {
            timings_ns,
            cpu_times_ns,
            warmup_timings_ns,
            alloc_bytes: avg_alloc,
        };

        let total_tokens = t * batch * seq;
        let params = format!(
            "{{\"thread_count\":{t},\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":{HIDDEN},\"workload\":\"train_step\"}}"
        );
        results.push(scenario_json(
            &format!("parallel_train_T{t}"),
            "parallel",
            &params,
            &m,
            0,
            0,
            None,
            None,
            Some(total_tokens),
            None,
        ));
    }
    results
}

// ─── Axis 5: Scale Comparison ──────────────────────────────────────────────

fn bench_scale_forward_256() -> String {
    eprintln!("  [scale] scale_forward_256");
    let batch = 2;
    let seq = 32;
    let mut model = MoETransformer::small();
    model.set_inference_mode(true);
    let ids = make_token_ids(batch, seq, VOCAB);
    let mut last_output = None;

    let m = measure(N_WARMUP, N_TRIALS, || {
        last_output = Some(model.forward_ids(&ids, batch, seq));
    });

    let (nan, inf, ma) = match &last_output {
        Some(t) => check_numerical(t),
        None => (0, 0, 0.0),
    };

    let params = format!("{{\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":256}}");
    scenario_json(
        "scale_forward_256",
        "scale",
        &params,
        &m,
        nan,
        inf,
        Some(ma),
        None,
        Some(batch * seq),
        None,
    )
}

fn bench_scale_train_256() -> String {
    eprintln!("  [scale] scale_train_256");
    let batch = 2;
    let seq = 8;
    let cfg = Config::small();
    let model = MoETransformer::new(cfg);
    let train_cfg = TrainConfig {
        batch_size: batch,
        seq_len: seq,
        lr: 1e-4,
        warmup_steps: 0,
        total_steps: 100,
        grad_clip: 1.0,
        aux_loss_weight: 0.01,
    };
    let mut trainer = Trainer::new(model, train_cfg);
    let input = make_input_tensor(batch, seq, VOCAB);
    let targets = make_targets_tensor(batch, seq, VOCAB);

    let m = measure(N_WARMUP, N_TRIALS, || {
        let _ = trainer.train_step(&input, &targets);
    });

    let params = format!("{{\"batch\":{batch},\"seq_len\":{seq},\"hidden_dim\":256}}");
    scenario_json(
        "scale_train_256",
        "scale",
        &params,
        &m,
        0,
        0,
        None,
        None,
        Some(batch * seq),
        None,
    )
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    eprintln!("Rust MoE Transformer Benchmark (5-axis, 22 scenarios)");
    eprintln!("=====================================================");

    let rust_version = match std::process::Command::new("rustc")
        .arg("--version")
        .output()
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        Err(_) => "unknown".into(),
    };
    let os_info = get_os_info();
    let cpu_model = get_cpu_model();
    let timestamp = get_timestamp();

    let mut scenario_jsons: Vec<String> = Vec::new();

    // Axis 1: Memory Management (9 scenarios)
    scenario_jsons.push(bench_mem_train_step());
    scenario_jsons.extend(bench_mem_scale_batch()); // 4
    scenario_jsons.extend(bench_mem_scale_seq()); // 4

    // Axis 2: Compiler Optimization (3 scenarios)
    scenario_jsons.push(bench_kernel_matmul());
    scenario_jsons.push(bench_kernel_softmax());
    scenario_jsons.push(bench_kernel_rmsnorm());

    // Axis 3: Type System (2 scenarios)
    scenario_jsons.push(bench_dispatch_warm());
    scenario_jsons.push(bench_dispatch_cold());

    // Axis 4: Parallel (3 scenarios)
    scenario_jsons.extend(bench_parallel());

    // Axis 4: Parallel Training (3 scenarios)
    scenario_jsons.extend(bench_parallel_train());

    // Axis 5: Scale Comparison (2 scenarios)
    scenario_jsons.push(bench_scale_forward_256());
    scenario_jsons.push(bench_scale_train_256());

    eprintln!("  done. {} scenarios collected.", scenario_jsons.len());

    let json = format!(
        concat!(
            "{{",
            "\"metadata\":{{",
            "\"language\":\"rust\",",
            "\"language_version\":\"{ver}\",",
            "\"os\":\"{os}\",",
            "\"cpu_model\":\"{cpu}\",",
            "\"timestamp\":\"{ts}\",",
            "\"n_trials\":{n_trials},",
            "\"n_warmup\":{n_warmup},",
            "\"seed\":{seed}",
            "}},",
            "\"scenarios\":[{scenarios}]",
            "}}"
        ),
        ver = escape_json_string(&rust_version),
        os = escape_json_string(&os_info),
        cpu = escape_json_string(&cpu_model),
        ts = escape_json_string(&timestamp),
        n_trials = N_TRIALS,
        n_warmup = N_WARMUP,
        seed = SEED,
        scenarios = scenario_jsons.join(","),
    );

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    if let Err(e) = handle.write_all(json.as_bytes()) {
        eprintln!("Failed to write JSON: {e}");
    }
    if let Err(e) = handle.write_all(b"\n") {
        eprintln!("Failed to write newline: {e}");
    }
}
