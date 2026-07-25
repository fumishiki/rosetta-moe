# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2025-2026 fumi-engineer

# bench_gpu.jl — Metal GPU benchmark harness for MoE Transformer
#
# Runs GPU-only scenarios using MetalTensor (MtlArray).
# Includes from src_gpu/ only (Metal.jl required).
#
# Usage: julia --project=. bench_gpu.jl

include("src_gpu/MoETransformerGPU.jl")
using .MoETransformerGPU
using Metal
using LinearAlgebra: mul!
using Random
using Dates
using Printf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
function _env_int(name::String, default::Int)::Int
    raw = get(ENV, name, string(default))
    parsed = try
        parse(Int, raw)
    catch
        default
    end
    parsed > 0 ? parsed : default
end

const N_TRIALS  = _env_int("ROSETTA_BENCH_TRIALS", 10)
const N_WARMUP  = _env_int("ROSETTA_BENCH_WARMUP", 3)
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
    "unknown"
end

function make_forward_input_mtl(batch::Int, seq_len::Int)
    data = Array{Float32}(undef, batch, seq_len)
    for b in 1:batch, s in 1:seq_len
        data[b, s] = Float32(((b - 1) * seq_len + (s - 1)) % VOCAB)
    end
    MetalTensor(MtlArray(data), F32)
end

function make_targets_mtl(batch::Int, seq_len::Int)
    data = Array{Float32}(undef, batch, seq_len)
    for b in 1:batch, s in 1:seq_len
        data[b, s] = Float32(((b - 1) * seq_len + (s - 1) + 1) % VOCAB)
    end
    MetalTensor(MtlArray(data), F32)
end

function check_numerical_mtl(t::MetalTensor)
    d = t.data
    nan_count = Int(round(sum(Float32.(isnan.(d)))))
    inf_count = Int(round(sum(Float32.(isinf.(d)))))
    max_abs = Float32(maximum(abs.(ifelse.(isfinite.(d), d, 0f0))))
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
    alloc_rate_bytes_per_sec::Float64
    gc_throughput::Float64
    gflops::Union{Float64,Nothing}
end

function run_scenario(;
    id::String,
    axis::String,
    params::Dict{String,Any},
    setup_fn::Function,
    run_fn::Function,
    n_warmup::Int = N_WARMUP,
    n_trials::Int = N_TRIALS,
    precompile::Bool = false,
    known_flops::Union{Int,Nothing} = nothing
)::ScenarioResult
    print(stderr, "  $(id)...")
    flush(stderr)

    if precompile
        pre_ctx = setup_fn()
        Metal.@sync run_fn(pre_ctx)
    end

    ctx = setup_fn()

    warmup_timings = Vector{UInt64}(undef, n_warmup)
    for i in 1:n_warmup
        tw0 = time_ns()
        Metal.@sync run_fn(ctx)
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
        result = Metal.@sync run_fn(ctx)
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
    if last_result isa MetalTensor
        nan_count, inf_count, max_abs = check_numerical_mtl(last_result)
    end

    b = get(params, "batch", 0)
    sl = get(params, "seq_len", 0)
    throughput = (b > 0 && sl > 0 && med > 0) ? Float64(b * sl) / (Float64(med) / 1e9) : 0.0

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

function main()
    if !metal_available()
        println(stderr, "ERROR: Metal GPU not available on this system")
        exit(1)
    end

    println(stderr, "Julia MoE Transformer GPU Benchmark (Metal)")
    println(stderr, "============================================================")
    set_inference_mode!(false)

    scenarios = ScenarioResult[]

    # GPU kernel benchmarks
    println(stderr, "GPU Kernels")

    # gpu_kernel_matmul
    push!(scenarios, run_scenario(
        id="gpu_kernel_matmul", axis="gpu",
        params=Dict{String,Any}("m" => 256, "n" => 256, "k" => 256),
        setup_fn=() -> begin
            a = MtlArray(randn(Float32, 256, 256))
            b = MtlArray(randn(Float32, 256, 256))
            c = MtlArray{Float32}(undef, 256, 256)
            (a=a, b=b, c=c)
        end,
        run_fn=ctx -> begin
            mul!(ctx.c, ctx.a, ctx.b)
            nothing
        end,
        precompile=true,
        known_flops=2*256*256*256
    ))

    # gpu_kernel_softmax
    push!(scenarios, run_scenario(
        id="gpu_kernel_softmax", axis="gpu",
        params=Dict{String,Any}("n" => 1000),
        setup_fn=() -> begin
            x = MtlArray(reshape(randn(Float32, 1000), 1, 1000))
            (x=x,)
        end,
        run_fn=ctx -> begin
            mx = maximum(ctx.x; dims=2)
            shifted = ctx.x .- mx
            e = exp.(shifted)
            s = sum(e; dims=2)
            _ = e ./ s
            nothing
        end,
        precompile=true,
        known_flops=4*1000
    ))

    # gpu_kernel_rmsnorm
    push!(scenarios, run_scenario(
        id="gpu_kernel_rmsnorm", axis="gpu",
        params=Dict{String,Any}("rows" => 2, "hidden_dim" => 64),
        setup_fn=() -> begin
            x = MtlArray(randn(Float32, 2, 64))
            w = MtlArray(ones(Float32, 64))
            (x=x, w=w)
        end,
        run_fn=ctx -> begin
            sum_sq = sum(ctx.x .* ctx.x; dims=2)
            inv_rms = 1f0 ./ sqrt.(sum_sq ./ 64f0 .+ 1f-6)
            _ = ctx.x .* inv_rms .* reshape(ctx.w, 1, 64)
            nothing
        end,
        precompile=true,
        known_flops=2*64*3
    ))

    # GPU forward benchmarks at different scales
    println(stderr, "GPU Forward")

    for (label, model_fn, hidden) in [("64", tiny_metal_model, 64), ("256", small_metal_model, 256), ("512", medium_metal_model, 512)]
        push!(scenarios, run_scenario(
            id="gpu_forward_$(label)", axis="gpu",
            params=Dict{String,Any}("batch" => 2, "seq_len" => 32, "hidden_dim" => hidden),
            setup_fn=() -> begin
                Random.seed!(SEED); seed_rng!(SEED)
                set_inference_mode!(true)
                model = model_fn()
                input = make_forward_input_mtl(2, 32)
                (model=model, input=input)
            end,
            run_fn=ctx -> begin
                gpu_forward(ctx.model, ctx.input)
                nothing
            end
        ))
    end

    # GPU train benchmarks at different scales
    println(stderr, "GPU Train")

    for (label, model_fn, hidden) in [("64", tiny_metal_model, 64), ("256", small_metal_model, 256), ("512", medium_metal_model, 512)]
        push!(scenarios, run_scenario(
            id="gpu_train_$(label)", axis="gpu",
            params=Dict{String,Any}("batch" => 2, "seq_len" => 8, "hidden_dim" => hidden),
            setup_fn=() -> begin
                Random.seed!(SEED); seed_rng!(SEED)
                set_inference_mode!(true)
                model = model_fn()
                cfg = default_metal_train_config()
                trainer = MetalTrainer(model, cfg)
                input = make_forward_input_mtl(2, 8)
                targets = make_targets_mtl(2, 8)
                (trainer=trainer, input=input, targets=targets)
            end,
            run_fn=ctx -> begin
                gpu_forward_ce_no_readback!(ctx.trainer, ctx.input, ctx.targets)
                nothing
            end
        ))
    end

    # JSON output
    cpu_model = get_cpu_model()
    timestamp = Dates.format(now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    scenarios_json = join(map(scenario_to_json, scenarios), ",\n    ")

    json = """{
  "metadata": {
    "language": "julia",
    "backend": "gpu_metal",
    "language_version": "$(escape_json(string(VERSION)))",
    "os": "$(escape_json(string(Sys.KERNEL)))",
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
    println(stderr, "GPU Benchmark complete. $(length(scenarios)) scenarios.")
    flush(stderr)
end

main()
