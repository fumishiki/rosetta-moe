# SPDX-License-Identifier: CC-BY-4.0
# Copyright (c) 2025-2026 fumi-engineer

# bench.jl — Benchmark harness for MoE Transformer
#
# Runs 22 scenarios across 5 axes:
#   1. Memory Management — train step and scaling (batch/seq)
#   2. Compiler Optimization — kernel benchmarks (matmul, softmax, rmsnorm)
#   3. Type System — dispatch_warm vs dispatch_cold (JIT compilation cost)
#   4. Parallel — multi-threaded forward pass scaling
#
# Outputs JSON to stdout for cross-language comparison.
# Uses getrusage(2) via ccall for CPU time measurement (user + system).

include("MoETransformer.jl")
using .MoETransformer
import .MoETransformer as MOE
using LinearAlgebra: mul!
using Random
using Dates
using Printf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
const N_TRIALS  = 10
const N_WARMUP  = 3
const SEED      = 42
const VOCAB     = 1000
const HIDDEN    = 64

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function get_cpu_model()
    if Sys.isapple()
        try return strip(read(`sysctl -n machdep.cpu.brand_string`, String)) catch; end
    end
    try
        for line in eachline("/proc/cpuinfo")
            startswith(line, "model name") && return strip(split(line, ":")[2])
        end
    catch; end
    "unknown"
end

function make_forward_input(batch::Int, seq_len::Int)
    data = Array{Float32}(undef, batch, seq_len)
    for b in 1:batch, s in 1:seq_len
        data[b, s] = Float32(((b - 1) * seq_len + (s - 1)) % VOCAB)
    end
    Tensor(data, F32)
end

function make_targets(batch::Int, seq_len::Int)
    data = Array{Float32}(undef, batch, seq_len)
    for b in 1:batch, s in 1:seq_len
        data[b, s] = Float32(((b - 1) * seq_len + (s - 1) + 1) % VOCAB)
    end
    Tensor(data, F32)
end

function check_numerical(t::Tensor)
    nan_count = count(isnan, t.data)
    inf_count = count(isinf, t.data)
    max_abs = isempty(t.data) ? 0.0f0 : mapreduce(x -> isfinite(x) ? abs(x) : 0.0f0, max, t.data; init=0.0f0)
    (nan_count, inf_count, max_abs)
end

function median_val(v::Vector{UInt64})
    s = sort(v)
    n = length(s)
    n == 0 && return UInt64(0)
    if n % 2 == 1
        s[div(n, 2) + 1]
    else
        div(s[div(n, 2)] + s[div(n, 2) + 1], UInt64(2))
    end
end

function p95_val(v::Vector{UInt64})
    s = sort(v)
    idx = Int(ceil(0.95 * length(s)))
    s[min(idx, length(s))]
end

function percentile_val(v::Vector{UInt64}, p::Float64)
    s = sort(v)
    n = length(s)
    k = (n - 1) * p / 100.0
    f = Int(floor(k)) + 1
    c = f + 1
    c > n && return s[n]
    UInt64(round(Float64(s[f]) + (k - (f - 1)) * Float64(s[c] - s[f])))
end

function iqr_val(v::Vector{UInt64})
    q1 = percentile_val(v, 25.0)
    q3 = percentile_val(v, 75.0)
    q3 - q1
end

function escape_json(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'; write(buf, "\\\"")
        elseif c == '\\'; write(buf, "\\\\")
        elseif c == '\n'; write(buf, "\\n")
        elseif c == '\r'; write(buf, "\\r")
        elseif c == '\t'; write(buf, "\\t")
        else write(buf, c)
        end
    end
    String(take!(buf))
end

format_timings(timings::Vector{UInt64}) = join(string.(timings), ", ")

# ---------------------------------------------------------------------------
# CPU time via getrusage(2) — user + system time in nanoseconds
# Uses ccall to invoke the C library function directly.
# struct rusage layout (macOS/Linux 64-bit): two timeval structs at offset 0
# and 16 respectively, each containing tv_sec (Int64) + tv_usec (Int64).
# ---------------------------------------------------------------------------
function get_cpu_time_ns()::UInt64
    buf = zeros(UInt8, 256)
    ret = ccall(:getrusage, Cint, (Cint, Ptr{UInt8}), 0, buf)
    ret != 0 && return UInt64(0)
    utime_sec  = reinterpret(Int64, buf[1:8])[1]
    utime_usec = reinterpret(Int64, buf[9:16])[1]
    stime_sec  = reinterpret(Int64, buf[17:24])[1]
    stime_usec = reinterpret(Int64, buf[25:32])[1]
    total_usec = UInt64(utime_sec * 1_000_000 + utime_usec + stime_sec * 1_000_000 + stime_usec)
    total_usec * UInt64(1000)
end

# ---------------------------------------------------------------------------
# Scenario result type
# ---------------------------------------------------------------------------
mutable struct ScenarioResult
    id::String
    axis::String
    params::Dict{String,Any}
    n_warmup::Int
    n_trials::Int
    timings_ns::Vector{UInt64}
    cpu_times_ns::Vector{UInt64}
    warmup_timings_ns::Vector{UInt64}
    median_ns::UInt64
    p95_ns::UInt64
    min_ns::UInt64
    max_ns::UInt64
    iqr_ns::UInt64
    cpu_median_ns::UInt64
    throughput_tokens_per_sec::Float64
    peak_rss_bytes::Int64
    alloc_bytes::Int64
    gc_total_time_ns::Int64
    gc_pause_count::Int64
    nan_count::Int
    inf_count::Int
    max_abs::Float32
    # derived
    alloc_rate_bytes_per_sec::Float64
    gc_throughput::Float64
    gflops::Union{Float64,Nothing}
end

# ---------------------------------------------------------------------------
# Generic scenario runner
# ---------------------------------------------------------------------------
function run_scenario(;
    id::String,
    axis::String,
    params::Dict{String,Any},
    setup_fn::Function,
    run_fn::Function,
    n_warmup::Int = N_WARMUP,
    n_trials::Int = N_TRIALS,
    known_flops::Union{Int,Nothing} = nothing
)::ScenarioResult
    print(stderr, "  $(id)...")
    flush(stderr)

    ctx = setup_fn()

    # Warmup: JIT compilation happens here (dispatch_cold measures this cost)
    warmup_timings = Vector{UInt64}(undef, n_warmup)
    for i in 1:n_warmup
        tw0 = time_ns()
        run_fn(ctx)
        tw1 = time_ns()
        warmup_timings[i] = UInt64(tw1 - tw0)
    end

    GC.gc()
    gc_before = Base.gc_num()

    timings = Vector{UInt64}(undef, n_trials)
    cpu_times = Vector{UInt64}(undef, n_trials)
    last_result = nothing

    for i in 1:n_trials
        cpu0 = get_cpu_time_ns()
        t0 = time_ns()
        result = run_fn(ctx)
        t1 = time_ns()
        cpu1 = get_cpu_time_ns()
        timings[i] = UInt64(t1 - t0)
        cpu_times[i] = cpu1 - cpu0
        if i == n_trials
            last_result = result
        end
    end

    gc_after = Base.gc_num()

    alloc_bytes = max(Int64(0), gc_after.allocd - gc_before.allocd)
    gc_time = gc_after.total_time - gc_before.total_time
    gc_pauses = gc_after.pause - gc_before.pause
    peak_rss = Sys.maxrss()

    med = n_trials > 0 ? median_val(timings) : UInt64(0)
    p95 = n_trials > 0 ? p95_val(timings) : UInt64(0)
    mn = n_trials > 0 ? minimum(timings) : UInt64(0)
    mx = n_trials > 0 ? maximum(timings) : UInt64(0)
    iqr = n_trials > 0 ? iqr_val(timings) : UInt64(0)
    cpu_med = n_trials > 0 ? median_val(cpu_times) : UInt64(0)

    nan_count = 0
    inf_count = 0
    max_abs = 0f0
    if last_result isa Tensor
        nan_count, inf_count, max_abs = check_numerical(last_result)
    end

    b = get(params, "batch", 0)
    sl = get(params, "seq_len", 0)
    tc = get(params, "thread_count", 1)
    throughput = (b > 0 && sl > 0 && med > 0) ? Float64(tc * b * sl) / (Float64(med) / 1e9) : 0.0

    alloc_rate = med > 0 ? Float64(alloc_bytes) / (Float64(med) * 1e-9) : 0.0
    sum_timings = Float64(sum(timings))
    gc_tp = sum_timings > 0 ? 1.0 - (Float64(gc_time) / sum_timings) : 1.0

    gflops = nothing
    if known_flops !== nothing && med > 0
        gflops = Float64(known_flops) / (Float64(med) * 1e-9) * 1e-9
    end

    println(stderr, " done (median=$(med)ns)")

    ScenarioResult(
        id, axis, params, n_warmup, n_trials,
        timings, cpu_times, warmup_timings,
        med, p95, mn, mx, iqr, cpu_med,
        throughput,
        peak_rss, alloc_bytes,
        gc_time, gc_pauses,
        nan_count, inf_count, max_abs,
        alloc_rate, gc_tp, gflops
    )
end

# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------
function scenario_to_json(s::ScenarioResult)
    param_entries = String[]
    for (k, v) in s.params
        if v isa String
            push!(param_entries, "\"$(escape_json(k))\": \"$(escape_json(v))\"")
        elseif v isa AbstractVector
            arr_str = join(string.(v), ", ")
            push!(param_entries, "\"$(escape_json(k))\": [$(arr_str)]")
        else
            push!(param_entries, "\"$(escape_json(k))\": $(v)")
        end
    end
    params_json = join(param_entries, ", ")

    timings_json = format_timings(s.timings_ns)
    cpu_json = format_timings(s.cpu_times_ns)
    warmup_json = format_timings(s.warmup_timings_ns)

    derived_entries = String[]
    push!(derived_entries, "\"alloc_rate_bytes_per_sec\": $(s.alloc_rate_bytes_per_sec)")
    push!(derived_entries, "\"gc_throughput\": $(s.gc_throughput)")
    if s.gflops !== nothing
        push!(derived_entries, "\"gflops\": $(s.gflops)")
    end
    derived_json = join(derived_entries, ", ")

    """{
      "id": "$(escape_json(s.id))",
      "axis": "$(escape_json(s.axis))",
      "params": {$(params_json)},
      "warmup_runs": $(s.n_warmup),
      "trial_runs": $(s.n_trials),
      "timings_ns": [$(timings_json)],
      "cpu_times_ns": [$(cpu_json)],
      "warmup_timings_ns": [$(warmup_json)],
      "median_ns": $(s.median_ns),
      "p95_ns": $(s.p95_ns),
      "min_ns": $(s.min_ns),
      "max_ns": $(s.max_ns),
      "iqr_ns": $(s.iqr_ns),
      "cpu_median_ns": $(s.cpu_median_ns),
      "throughput_tokens_per_sec": $(s.throughput_tokens_per_sec),
      "memory": {"peak_rss_bytes": $(s.peak_rss_bytes), "alloc_bytes": $(s.alloc_bytes)},
      "gc": {"total_gc_time_ns": $(s.gc_total_time_ns), "gc_pause_count": $(s.gc_pause_count)},
      "numerical": {"nan_count": $(s.nan_count), "inf_count": $(s.inf_count), "max_abs": $(s.max_abs)},
      "derived": {$(derived_json)}
    }"""
end

# ---------------------------------------------------------------------------
# Main — runs all 22 scenarios and outputs JSON
# ---------------------------------------------------------------------------
function main()
    println(stderr, "Julia MoE Transformer Benchmark (5-axis, 22 scenarios)")
    println(stderr, "========================================================")

    scenarios = ScenarioResult[]

    # ===================================================================
    # Axis 1: Memory Management (axis="memory")
    # ===================================================================
    println(stderr, "Axis 1: Memory Management")

    # mem_train_step
    push!(scenarios, run_scenario(
        id="mem_train_step", axis="memory",
        params=Dict{String,Any}("batch" => 2, "seq_len" => 8, "hidden_dim" => 64),
        setup_fn=() -> begin
            Random.seed!(SEED)
            m = tiny_model()
            cfg = default_train_config()
            trainer = Trainer(m, cfg)
            input = make_forward_input(2, 8)
            targets = make_targets(2, 8)
            (trainer=trainer, input=input, targets=targets)
        end,
        run_fn=ctx -> train_step!(ctx.trainer, ctx.input, ctx.targets)
    ))

    # mem_scale_batch_{1,2,4,8}
    for bs in [1, 2, 4, 8]
        push!(scenarios, run_scenario(
            id="mem_scale_batch_$(bs)", axis="memory",
            params=Dict{String,Any}("batch" => bs, "seq_len" => 32, "hidden_dim" => 64),
            setup_fn=() -> begin
                Random.seed!(SEED)
                m = tiny_model()
                input = make_forward_input(bs, 32)
                (model=m, input=input)
            end,
            run_fn=ctx -> forward(ctx.model, ctx.input)
        ))
    end

    # mem_scale_seq_{8,16,32,64}
    for seq in [8, 16, 32, 64]
        push!(scenarios, run_scenario(
            id="mem_scale_seq_$(seq)", axis="memory",
            params=Dict{String,Any}("batch" => 2, "seq_len" => seq, "hidden_dim" => 64),
            setup_fn=() -> begin
                Random.seed!(SEED)
                m = tiny_model()
                input = make_forward_input(2, seq)
                (model=m, input=input)
            end,
            run_fn=ctx -> forward(ctx.model, ctx.input)
        ))
    end

    # ===================================================================
    # Axis 2: Compiler Optimization (axis="compiler")
    # Measures raw kernel performance (BLAS, softmax, rmsnorm)
    # ===================================================================
    println(stderr, "Axis 2: Compiler Optimization")

    # kernel_matmul: M=K=N=64, known_flops=2*64*64*64=524288
    # Pre-allocate output and materialize transposed b so we time only mul! (BLAS sgemm)
    push!(scenarios, run_scenario(
        id="kernel_matmul", axis="compiler",
        params=Dict{String,Any}("M" => 64, "K" => 64, "N" => 64),
        setup_fn=() -> begin
            Random.seed!(SEED)
            a = randn(Float32, 64, 64)
            b = randn(Float32, 64, 64)
            bt = collect(transpose(b))
            out = Matrix{Float32}(undef, 64, 64)
            (a=a, bt=bt, out=out)
        end,
        run_fn=ctx -> begin
            mul!(ctx.out, ctx.a, ctx.bt)
            Tensor(ctx.out, F32)
        end,
        known_flops=524288
    ))

    # kernel_softmax: n=1000, known_flops=4000
    push!(scenarios, run_scenario(
        id="kernel_softmax", axis="compiler",
        params=Dict{String,Any}("n" => 1000),
        setup_fn=() -> begin
            Random.seed!(SEED)
            v = randn(Float32, 1000)
            buf = similar(v)
            (v=v, buf=buf)
        end,
        run_fn=ctx -> begin
            copyto!(ctx.buf, ctx.v)
            softmax_in_place!(ctx.buf)
            Tensor(reshape(ctx.buf, 1, 1000), F32)
        end,
        known_flops=4000
    ))

    # kernel_rmsnorm: shape=(2,32,64), known_flops=2*32*64*3=12288
    push!(scenarios, run_scenario(
        id="kernel_rmsnorm", axis="compiler",
        params=Dict{String,Any}("shape" => [2, 32, 64]),
        setup_fn=() -> begin
            Random.seed!(SEED)
            t = randn_tensor(2, 32, 64)
            norm = RMSNorm(64)
            (t=t, norm=norm)
        end,
        run_fn=ctx -> forward(ctx.norm, ctx.t),
        known_flops=12288
    ))

    # ===================================================================
    # Axis 3: Type System (axis="type_system")
    # dispatch_warm: steady-state forward (JIT already compiled)
    # dispatch_cold: new model + first forward each trial (measures JIT cost)
    # ===================================================================
    println(stderr, "Axis 3: Type System")

    push!(scenarios, run_scenario(
        id="dispatch_warm", axis="type_system",
        params=Dict{String,Any}("batch" => 2, "seq_len" => 32, "hidden_dim" => 64),
        setup_fn=() -> begin
            Random.seed!(SEED)
            m = tiny_model()
            input = make_forward_input(2, 32)
            (model=m, input=input)
        end,
        run_fn=ctx -> forward(ctx.model, ctx.input)
    ))

    # dispatch_cold: construct NEW model + first forward each trial, n_warmup=0
    # This measures JIT compilation overhead for first-time method specialization.
    push!(scenarios, run_scenario(
        id="dispatch_cold", axis="type_system",
        params=Dict{String,Any}("batch" => 1, "seq_len" => 8, "hidden_dim" => 64),
        n_warmup=N_WARMUP,
        setup_fn=() -> begin
            Random.seed!(SEED)
            input = make_forward_input(1, 8)
            (input=input,)
        end,
        run_fn=ctx -> begin
            m = tiny_model()
            forward(m, ctx.input)
        end
    ))

    # ===================================================================
    # Axis 4: Parallel (axis="parallel")
    # Tests multi-threaded forward pass scaling with Threads.@threads.
    # JULIA_NUM_THREADS must be set before Julia launch (cannot change at runtime).
    # ===================================================================
    println(stderr, "Axis 4: Parallel")

    available_threads = Threads.nthreads()

    for T in [1, 2, 4]
        batch_pt = 2
        seq_pt = 32
        # Capture T in a local for the closure (avoids boxing)
        local nT = T
        push!(scenarios, run_scenario(
            id="parallel_T$(nT)", axis="parallel",
            params=Dict{String,Any}(
                "thread_count" => nT,
                "available_threads" => available_threads,
                "batch" => batch_pt, "seq_len" => seq_pt, "hidden_dim" => 64
            ),
            setup_fn=() -> begin
                Random.seed!(SEED)
                # Each thread gets its own independent model instance (no shared mutable state)
                models = [tiny_model() for _ in 1:nT]
                inputs = [make_forward_input(batch_pt, seq_pt) for _ in 1:nT]
                results = Vector{Tensor}(undef, nT)
                (models=models, inputs=inputs, results=results, T=nT)
            end,
            run_fn=ctx -> begin
                Threads.@threads for i in 1:ctx.T
                    ctx.results[i] = forward(ctx.models[i], ctx.inputs[i])
                end
                ctx.results[1]
            end
        ))
    end

    # parallel_train_T{1,2,4}: multi-threaded training step scaling
    for T in [1, 2, 4]
        batch_pt = 2
        seq_pt = 8
        local nT = T
        push!(scenarios, run_scenario(
            id="parallel_train_T$(nT)", axis="parallel",
            params=Dict{String,Any}(
                "thread_count" => nT,
                "available_threads" => available_threads,
                "batch" => batch_pt, "seq_len" => seq_pt, "hidden_dim" => 64,
                "workload" => "train_step"
            ),
            setup_fn=() -> begin
                Random.seed!(SEED)
                models = [tiny_model() for _ in 1:nT]
                cfgs = [default_train_config() for _ in 1:nT]
                trainers = [Trainer(models[i], cfgs[i]) for i in 1:nT]
                inputs = [make_forward_input(batch_pt, seq_pt) for _ in 1:nT]
                targets_list = [make_targets(batch_pt, seq_pt) for _ in 1:nT]
                losses = Vector{Float32}(undef, nT)
                (trainers=trainers, inputs=inputs, targets=targets_list, losses=losses, T=nT)
            end,
            run_fn=ctx -> begin
                Threads.@threads for i in 1:ctx.T
                    ctx.losses[i] = train_step!(ctx.trainers[i], ctx.inputs[i], ctx.targets[i])
                end
                ctx.losses[1]
            end
        ))
    end

    # ===================================================================
    # Axis 5: Scale Comparison (axis="scale")
    # Tests how performance gaps change at hidden=256 vs hidden=64.
    # ===================================================================
    println(stderr, "Axis 5: Scale Comparison")

    push!(scenarios, run_scenario(
        id="scale_forward_256", axis="scale",
        params=Dict{String,Any}("batch" => 2, "seq_len" => 32, "hidden_dim" => 256),
        setup_fn=() -> begin
            Random.seed!(SEED)
            m = small_model()
            input = make_forward_input(2, 32)
            (model=m, input=input)
        end,
        run_fn=ctx -> forward(ctx.model, ctx.input)
    ))

    push!(scenarios, run_scenario(
        id="scale_train_256", axis="scale",
        params=Dict{String,Any}("batch" => 2, "seq_len" => 8, "hidden_dim" => 256),
        setup_fn=() -> begin
            Random.seed!(SEED)
            m = small_model()
            cfg = default_train_config()
            trainer = Trainer(m, cfg)
            input = make_forward_input(2, 8)
            targets = make_targets(2, 8)
            (trainer=trainer, input=input, targets=targets)
        end,
        run_fn=ctx -> train_step!(ctx.trainer, ctx.input, ctx.targets)
    ))

    # ===================================================================
    # JSON output
    # ===================================================================
    cpu_model = get_cpu_model()
    timestamp = Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    os_name = string(Sys.KERNEL)
    julia_version = string(VERSION)

    scenarios_json = join(map(scenario_to_json, scenarios), ",\n    ")

    json = """{
  "metadata": {
    "language": "julia",
    "language_version": "$(escape_json(julia_version))",
    "os": "$(escape_json(os_name))",
    "cpu_model": "$(escape_json(cpu_model))",
    "timestamp": "$(timestamp)",
    "n_trials": $(N_TRIALS),
    "n_warmup": $(N_WARMUP),
    "seed": $(SEED)
  },
  "scenarios": [
    $(scenarios_json)
  ]
}
"""

    print(json)
    flush(stdout)

    println(stderr, "Benchmark complete. $(length(scenarios)) scenarios.")
    flush(stderr)
end

main()
