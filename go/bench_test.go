// SPDX-License-Identifier: CC-BY-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// Benchmark harness for the MoE Transformer.
//
// Measures performance across four axes:
//   1. Memory management — allocation rate, GC throughput under ML workload
//   2. Compiler optimization — kernel-level ops (matmul, softmax, rmsnorm)
//   3. Type system / dispatch — warm vs cold model dispatch overhead
//   4. Parallelism — goroutine scaling with independent models + WaitGroup
//
// Output: JSON to stdout, consumed by the cross-language comparison dashboard.
// Each scenario records wall time, CPU time, peak RSS, GC stats, and
// numerical sanity checks (NaN/Inf counts, max absolute value).

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"syscall"
	"testing"
	"time"
)

const (
	benchSeed    = 42
	benchNTrials = 10
	benchNWarmup = 3
	benchVocab   = 1000
	benchHidden  = 64
)

// --- JSON schema types ---

type benchMetadata struct {
	Language        string `json:"language"`
	LanguageVersion string `json:"language_version"`
	OS              string `json:"os"`
	CPUModel        string `json:"cpu_model"`
	Timestamp       string `json:"timestamp"`
	NTrials         int    `json:"n_trials"`
	NWarmup         int    `json:"n_warmup"`
	Seed            int    `json:"seed"`
}

type benchMemoryInfo struct {
	PeakRSSBytes uint64 `json:"peak_rss_bytes"`
	AllocBytes   uint64 `json:"alloc_bytes"`
}

type benchGCInfo struct {
	TotalGCTimeNS int64 `json:"total_gc_time_ns"`
	GCPauseCount  int64 `json:"gc_pause_count"`
}

type benchNumerical struct {
	NaNCount int     `json:"nan_count"`
	InfCount int     `json:"inf_count"`
	MaxAbs   float32 `json:"max_abs"`
}

type benchDerived struct {
	AllocRateBytesPerSec float64 `json:"alloc_rate_bytes_per_sec"`
	GCThroughput         float64 `json:"gc_throughput"`
	GFLOPS               float64 `json:"gflops,omitempty"`
}

type benchScenario struct {
	ID                     string            `json:"id"`
	Axis                   string            `json:"axis"`
	Params                 map[string]any    `json:"params"`
	WarmupRuns             int               `json:"warmup_runs"`
	TrialRuns              int               `json:"trial_runs"`
	TimingsNS              []int64           `json:"timings_ns"`
	CPUTimesNS             []int64           `json:"cpu_times_ns"`
	WarmupTimingsNS        []int64           `json:"warmup_timings_ns"`
	MedianNS               int64             `json:"median_ns"`
	P95NS                  int64             `json:"p95_ns"`
	MinNS                  int64             `json:"min_ns"`
	MaxNS                  int64             `json:"max_ns"`
	IqrNS                  int64             `json:"iqr_ns"`
	CPUMedianNS            int64             `json:"cpu_median_ns"`
	ThroughputTokensPerSec float64           `json:"throughput_tokens_per_sec"`
	Memory                 benchMemoryInfo   `json:"memory"`
	GC                     benchGCInfo       `json:"gc"`
	Numerical              benchNumerical    `json:"numerical"`
	Derived                benchDerived      `json:"derived"`
}

type benchResult struct {
	Metadata  benchMetadata   `json:"metadata"`
	Scenarios []benchScenario `json:"scenarios"`
}

// --- helpers ---

// cpuModel reads the CPU brand string via sysctl (macOS-specific).
func cpuModel() string {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
	if err != nil {
		return fmt.Sprintf("%d cores", runtime.NumCPU())
	}
	s := string(out)
	if len(s) > 0 && s[len(s)-1] == '\n' {
		s = s[:len(s)-1]
	}
	return s
}

func medianInt64(sorted []int64) int64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func percentileInt64(sorted []int64, p float64) int64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	k := float64(n-1) * p / 100.0
	f := int(math.Floor(k))
	c := f + 1
	if c >= n {
		return sorted[n-1]
	}
	lower := float64(sorted[f])
	upper := float64(sorted[c])
	return int64(lower + (k-float64(f))*(upper-lower))
}

func iqrInt64(sorted []int64) int64 {
	return percentileInt64(sorted, 75.0) - percentileInt64(sorted, 25.0)
}

// getPeakRSSBytes returns peak resident set size via getrusage.
// On macOS, Maxrss is already in bytes; on Linux it's in kilobytes.
func getPeakRSSBytes() uint64 {
	var rusage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &rusage); err != nil {
		return 0
	}
	if runtime.GOOS == "darwin" {
		return uint64(rusage.Maxrss) // bytes on macOS
	}
	return uint64(rusage.Maxrss) * 1024
}

// getCPUTimeNS returns total user+system CPU time in nanoseconds.
func getCPUTimeNS() int64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	return usage.Utime.Nano() + usage.Stime.Nano()
}

// countNaNInf counts NaN and Inf values in the output for numerical sanity.
// Uses the IEEE 754 property: NaN != NaN.
func countNaNInf(data []float32) (nanCount, infCount int) {
	for _, v := range data {
		if v != v {
			nanCount++
		} else if v > math.MaxFloat32 || v < -math.MaxFloat32 {
			infCount++
		}
	}
	return
}

func maxAbsF32(data []float32) float32 {
	m := float32(0)
	for _, v := range data {
		a := float32(math.Abs(float64(v)))
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) && a > m {
			m = a
		}
	}
	return m
}

// paramInt extracts an integer parameter from the params map with type assertion.
func paramInt(params map[string]any, key string) (int, bool) {
	v, ok := params[key]
	if !ok {
		return 0, false
	}
	n, ok := v.(int)
	return n, ok
}

// makeInput creates deterministic token ID tensors for benchmarking.
func makeInput(batch, seqLen int) *Tensor {
	data := make([]float32, batch*seqLen)
	for i := range data {
		data[i] = float32(i % benchVocab)
	}
	return FromSlice(data, NewShape(batch, seqLen))
}

func makeTargets(batch, seqLen int) *Tensor {
	data := make([]float32, batch*seqLen)
	for i := range data {
		data[i] = float32((i + 1) % benchVocab)
	}
	return FromSlice(data, NewShape(batch, seqLen))
}

func makeRandTensor(rng *rand.Rand, dims ...int) *Tensor {
	shape := NewShape(dims...)
	t := New(shape, F32)
	for i := range t.DataPtr() {
		t.DataPtr()[i] = float32(rng.NormFloat64())
	}
	return t
}

// trialResult captures all measured data for a scenario.
type trialResult struct {
	timings       []int64
	cpuTimes      []int64
	warmupTimings []int64
	mem           benchMemoryInfo
	gc            benchGCInfo
	num           benchNumerical
}

// runTrials executes warmup + timed trials, collecting wall/cpu/mem/gc/numerical metrics.
// Memory and GC stats are measured only across the timed trials (not warmup) to get
// a per-trial allocation rate without warmup pollution.
func runTrials(warmup, trials int, setup func(), run func() []float32) trialResult {
	setup()

	warmupTimings := make([]int64, warmup)
	for i := 0; i < warmup; i++ {
		start := time.Now()
		run()
		warmupTimings[i] = time.Since(start).Nanoseconds()
	}

	// Snapshot mem/gc before trials to measure delta
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)
	var gcBefore debug.GCStats
	debug.ReadGCStats(&gcBefore)

	timings := make([]int64, trials)
	cpuTimes := make([]int64, trials)
	var lastOutput []float32
	for i := 0; i < trials; i++ {
		cpuBefore := getCPUTimeNS()
		start := time.Now()
		lastOutput = run()
		timings[i] = time.Since(start).Nanoseconds()
		cpuTimes[i] = getCPUTimeNS() - cpuBefore
	}

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	var gcAfter debug.GCStats
	debug.ReadGCStats(&gcAfter)

	nanCount, infCount := 0, 0
	ma := float32(0)
	if lastOutput != nil {
		nanCount, infCount = countNaNInf(lastOutput)
		ma = maxAbsF32(lastOutput)
	}

	gcTimeNS := int64(gcAfter.PauseTotal - gcBefore.PauseTotal)

	return trialResult{
		timings:       timings,
		cpuTimes:      cpuTimes,
		warmupTimings: warmupTimings,
		mem: benchMemoryInfo{
			PeakRSSBytes: getPeakRSSBytes(),
			AllocBytes:   memAfter.TotalAlloc - memBefore.TotalAlloc,
		},
		gc: benchGCInfo{
			TotalGCTimeNS: gcTimeNS,
			GCPauseCount:  gcAfter.NumGC - gcBefore.NumGC,
		},
		num: benchNumerical{NaNCount: nanCount, InfCount: infCount, MaxAbs: ma},
	}
}

// buildScenario computes statistics and derived metrics from raw trial results.
//
// Derived metrics:
//   alloc_rate = total_alloc_bytes / num_trials / median_wall_sec
//   gc_throughput = 1.0 - (gc_time / sum_of_all_trial_times)  -- fraction of time NOT in GC
//   gflops = known_flops / median_wall_sec / 1e9              -- if FLOP count is provided
func buildScenario(id, axis string, params map[string]any, warmupCount, trialCount int, tr trialResult, knownFlops float64) benchScenario {
	sorted := make([]int64, len(tr.timings))
	copy(sorted, tr.timings)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	medNS := medianInt64(sorted)

	sortedCPU := make([]int64, len(tr.cpuTimes))
	copy(sortedCPU, tr.cpuTimes)
	sort.Slice(sortedCPU, func(i, j int) bool { return sortedCPU[i] < sortedCPU[j] })
	cpuMedian := medianInt64(sortedCPU)

	// Throughput: tokens/sec computed from batch*seq_len / wall_time
	throughput := float64(0)
	batch, hasBatch := paramInt(params, "batch")
	seqLen, hasSeq := paramInt(params, "seq_len")
	threadCount, hasThread := paramInt(params, "thread_count")
	if medNS > 0 {
		wallSec := float64(medNS) / 1e9
		switch {
		case hasThread && hasBatch && hasSeq:
			throughput = float64(threadCount*batch*seqLen) / wallSec
		case hasBatch && hasSeq:
			throughput = float64(batch*seqLen) / wallSec
		}
	}

	var derived benchDerived
	if medNS > 0 {
		medSec := float64(medNS) / 1e9
		derived.AllocRateBytesPerSec = float64(tr.mem.AllocBytes) / float64(trialCount) / medSec

		// gc_throughput = 1.0 - (gc_time / sum_timings)
		sumTimingsNS := int64(0)
		for _, t := range tr.timings {
			sumTimingsNS += t
		}
		if sumTimingsNS > 0 && tr.gc.TotalGCTimeNS > 0 {
			derived.GCThroughput = 1.0 - float64(tr.gc.TotalGCTimeNS)/float64(sumTimingsNS)
		} else {
			derived.GCThroughput = 1.0
		}

		if knownFlops > 0 {
			derived.GFLOPS = knownFlops / medSec / 1e9
		}
	}

	minNS := int64(0)
	maxNS := int64(0)
	if len(sorted) > 0 {
		minNS = sorted[0]
		maxNS = sorted[len(sorted)-1]
	}

	return benchScenario{
		ID:                     id,
		Axis:                   axis,
		Params:                 params,
		WarmupRuns:             warmupCount,
		TrialRuns:              trialCount,
		TimingsNS:              tr.timings,
		CPUTimesNS:             tr.cpuTimes,
		WarmupTimingsNS:        tr.warmupTimings,
		MedianNS:               medNS,
		P95NS:                  percentileInt64(sorted, 95.0),
		MinNS:                  minNS,
		MaxNS:                  maxNS,
		IqrNS:                  iqrInt64(sorted),
		CPUMedianNS:            cpuMedian,
		ThroughputTokensPerSec: throughput,
		Memory:                 tr.mem,
		GC:                     tr.gc,
		Numerical:              tr.num,
		Derived:                derived,
	}
}

// --- TestBench: main entry point ---
//
// Runs as a Go test (not a benchmark) because we need full control over
// trial execution, GC measurement, and JSON output format. Standard Go
// benchmarks (testing.B) don't support the multi-metric collection needed here.

func TestBench(t *testing.T) {
	rand.Seed(benchSeed)

	result := benchResult{
		Metadata: benchMetadata{
			Language:        "go",
			LanguageVersion: runtime.Version(),
			OS:              runtime.GOOS,
			CPUModel:        cpuModel(),
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
			NTrials:         benchNTrials,
			NWarmup:         benchNWarmup,
			Seed:            benchSeed,
		},
	}

	cfg := Tiny()

	// =====================================================================
	// Axis 1: Memory Management (axis="memory")
	// =====================================================================

	// 1. mem_train_step: batch=2, seq=8, hidden=64 -- 1 train step
	{
		model := NewMoETransformer(cfg)
		trainer := NewTrainer(model, DefaultTrainConfig())
		input := makeInput(2, 8)
		targets := makeTargets(2, 8)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			loss := trainer.TrainStep(input, targets)
			return []float32{loss}
		})
		result.Scenarios = append(result.Scenarios, buildScenario("mem_train_step", "memory",
			map[string]any{"batch": 2, "seq_len": 8, "hidden_dim": benchHidden},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// 2-5. mem_scale_batch_{1,2,4,8}: seq=32, hidden=64 -- full forward
	for _, batchSize := range []int{1, 2, 4, 8} {
		model := NewMoETransformer(cfg)
		input := makeInput(batchSize, 32)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := model.Forward(input)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario(
			fmt.Sprintf("mem_scale_batch_%d", batchSize), "memory",
			map[string]any{"batch": batchSize, "seq_len": 32, "hidden_dim": benchHidden},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// 6-9. mem_scale_seq_{8,16,32,64}: batch=2, hidden=64 -- full forward
	for _, seqLen := range []int{8, 16, 32, 64} {
		model := NewMoETransformer(cfg)
		input := makeInput(2, seqLen)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := model.Forward(input)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario(
			fmt.Sprintf("mem_scale_seq_%d", seqLen), "memory",
			map[string]any{"batch": 2, "seq_len": seqLen, "hidden_dim": benchHidden},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// =====================================================================
	// Axis 2: Compiler Optimization (axis="compiler")
	// =====================================================================

	rng := rand.New(rand.NewSource(benchSeed))

	// 10. kernel_matmul: M=K=N=64, known_flops = 2*M*N*K = 524288
	{
		a := makeRandTensor(rng, 64, 64)
		b := makeRandTensor(rng, 64, 64)
		out := make([]float32, 64*64)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			sgemm(64, 64, 64, 1.0, a.DataPtr(), 64, b.DataPtr(), 64, 0.0, out, 64)
			return out
		})
		result.Scenarios = append(result.Scenarios, buildScenario("kernel_matmul", "compiler",
			map[string]any{"M": 64, "K": 64, "N": 64},
			benchNWarmup, benchNTrials, tr, 524288))
	}

	// 11. kernel_softmax: n=1000, known_flops = 4*n = 4000
	{
		t := makeRandTensor(rng, 1, 1000)
		out := New(NewShape(1, 1000), F32)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			t.SoftmaxInto(out)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario("kernel_softmax", "compiler",
			map[string]any{"n": 1000},
			benchNWarmup, benchNTrials, tr, 4000))
	}

	// 12. kernel_rmsnorm: shape=[2,32,64], known_flops = 3*numel = 12288
	{
		normInput := makeRandTensor(rng, 2, 32, 64)
		norm := NewRMSNorm(64, 1e-6)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := norm.Forward(normInput)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario("kernel_rmsnorm", "compiler",
			map[string]any{"shape": []int{2, 32, 64}},
			benchNWarmup, benchNTrials, tr, 12288))
	}

	// =====================================================================
	// Axis 3: Type System (axis="type_system")
	// =====================================================================

	// 13. dispatch_warm: reuse model, measure steady-state dispatch
	{
		model := NewMoETransformer(cfg)
		input := makeInput(2, 32)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := model.Forward(input)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario("dispatch_warm", "type_system",
			map[string]any{"batch": 2, "seq_len": 32, "hidden_dim": benchHidden},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// 14. dispatch_cold: construct NEW model + first forward per trial, no warmup
	{
		input := makeInput(1, 8)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			model := NewMoETransformer(cfg)
			out := model.Forward(input)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario("dispatch_cold", "type_system",
			map[string]any{"batch": 1, "seq_len": 8, "hidden_dim": benchHidden},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// =====================================================================
	// Axis 4: Parallel (axis="parallel")
	// =====================================================================

	// 15-17. parallel_T{1,2,4}: goroutines with independent models + WaitGroup.
	// Each goroutine owns its own model -- no shared mutable state, no locks.
	// This tests Go's goroutine scheduling overhead and GC behavior under
	// parallel allocation pressure.
	for _, T := range []int{1, 2, 4} {
		threadCount := T
		inputs := make([]*Tensor, threadCount)
		for i := range inputs {
			inputs[i] = makeInput(2, 32)
		}
		// Pre-create one model per goroutine OUTSIDE the timing loop
		models := make([]*MoETransformer, threadCount)
		for i := range models {
			models[i] = NewMoETransformer(cfg)
		}

		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			var wg sync.WaitGroup
			wg.Add(threadCount)
			for g := 0; g < threadCount; g++ {
				inp := inputs[g]
				m := models[g]
				go func() {
					defer wg.Done()
					m.Forward(inp)
				}()
			}
			wg.Wait()
			return nil
		})

		result.Scenarios = append(result.Scenarios, buildScenario(
			fmt.Sprintf("parallel_T%d", threadCount), "parallel",
			map[string]any{"batch": 2, "seq_len": 32, "hidden_dim": benchHidden, "thread_count": threadCount},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// 18-20. parallel_train_T{1,2,4}: goroutines with independent trainers + WaitGroup.
	// Each goroutine owns its own trainer -- no shared mutable state.
	// This tests training throughput scaling under parallel allocation pressure.
	for _, T := range []int{1, 2, 4} {
		threadCount := T
		inputs := make([]*Tensor, threadCount)
		targetsList := make([]*Tensor, threadCount)
		for i := range inputs {
			inputs[i] = makeInput(2, 8)
			targetsList[i] = makeTargets(2, 8)
		}
		trainers := make([]*Trainer, threadCount)
		for i := range trainers {
			trainers[i] = NewTrainer(NewMoETransformer(cfg), DefaultTrainConfig())
		}

		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			var wg sync.WaitGroup
			wg.Add(threadCount)
			for g := 0; g < threadCount; g++ {
				inp := inputs[g]
				tgt := targetsList[g]
				trainer := trainers[g]
				go func() {
					defer wg.Done()
					trainer.TrainStep(inp, tgt)
				}()
			}
			wg.Wait()
			return nil
		})

		result.Scenarios = append(result.Scenarios, buildScenario(
			fmt.Sprintf("parallel_train_T%d", threadCount), "parallel",
			map[string]any{"batch": 2, "seq_len": 8, "hidden_dim": benchHidden, "thread_count": threadCount, "workload": "train_step"},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// =====================================================================
	// Axis 5: Scale Comparison (axis="scale")
	// =====================================================================

	smallCfg := Small()

	// scale_forward_256: forward pass with hidden=256
	{
		model := NewMoETransformer(smallCfg)
		input := makeInput(2, 32)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := model.Forward(input)
			return out.DataPtr()
		})
		result.Scenarios = append(result.Scenarios, buildScenario("scale_forward_256", "scale",
			map[string]any{"batch": 2, "seq_len": 32, "hidden_dim": 256},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// scale_train_256: training step with hidden=256
	{
		model := NewMoETransformer(smallCfg)
		trainer := NewTrainer(model, DefaultTrainConfig())
		input := makeInput(2, 8)
		targets := makeTargets(2, 8)
		tr := runTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			loss := trainer.TrainStep(input, targets)
			return []float32{loss}
		})
		result.Scenarios = append(result.Scenarios, buildScenario("scale_train_256", "scale",
			map[string]any{"batch": 2, "seq_len": 8, "hidden_dim": 256},
			benchNWarmup, benchNTrials, tr, 0))
	}

	// =====================================================================
	// Output JSON
	// =====================================================================

	out, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal JSON: %v", err)
	}
	fmt.Fprintln(os.Stdout, string(out))
}
