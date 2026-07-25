// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build darwin

package gpu

// Benchmark harness for GPU operations (Metal/MPS).
//
// Measures performance across GPU-specific scenarios:
//   - gpu_kernel_matmul: MPS GEMM 256x256
//   - gpu_kernel_softmax: GPU row-wise softmax kernel
//   - gpu_kernel_rmsnorm: GPU RMSNorm kernel
//   - gpu_forward_{64,256,512}: full GPU forward pass
//   - gpu_train_{64,256,512}: full GPU forward + CE loss sum (no readback)
//
// Output: JSON to stdout, consumed by the cross-language comparison dashboard.

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"syscall"
	"testing"
	"time"
)

const (
	benchSeed = 42
)

var (
	benchNTrials = benchEnvInt("ROSETTA_BENCH_TRIALS", 10)
	benchNWarmup = benchEnvInt("ROSETTA_BENCH_WARMUP", 3)
)

func benchEnvInt(name string, def int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return def
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return def
	}
	return v
}

// --- JSON schema types ---

type benchMetadata struct {
	Language        string `json:"language"`
	Backend         string `json:"backend"`
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
	ID                     string          `json:"id"`
	Axis                   string          `json:"axis"`
	Params                 map[string]any  `json:"params"`
	WarmupRuns             int             `json:"warmup_runs"`
	TrialRuns              int             `json:"trial_runs"`
	TimingsNS              []int64         `json:"timings_ns"`
	CPUTimesNS             []int64         `json:"cpu_times_ns"`
	WarmupTimingsNS        []int64         `json:"warmup_timings_ns"`
	MedianNS               int64           `json:"median_ns"`
	P95NS                  int64           `json:"p95_ns"`
	MinNS                  int64           `json:"min_ns"`
	MaxNS                  int64           `json:"max_ns"`
	IqrNS                  int64           `json:"iqr_ns"`
	CPUMedianNS            int64           `json:"cpu_median_ns"`
	ThroughputTokensPerSec float64         `json:"throughput_tokens_per_sec"`
	Memory                 benchMemoryInfo `json:"memory"`
	GC                     benchGCInfo     `json:"gc"`
	Numerical              benchNumerical  `json:"numerical"`
	Derived                benchDerived    `json:"derived"`
}

type benchResult struct {
	Metadata  benchMetadata   `json:"metadata"`
	Scenarios []benchScenario `json:"scenarios"`
}

// --- helpers ---

func benchGpuCPUModel() string {
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

func benchGPUShaderDir() string {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return filepath.Clean(filepath.Join("..", "..", "shaders"))
	}
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", "shaders"))
}

func gpuMedianInt64(sorted []int64) int64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func gpuPercentileInt64(sorted []int64, p float64) int64 {
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

func gpuIqrInt64(sorted []int64) int64 {
	return gpuPercentileInt64(sorted, 75.0) - gpuPercentileInt64(sorted, 25.0)
}

func gpuGetPeakRSSBytes() uint64 {
	var rusage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &rusage); err != nil {
		return 0
	}
	if runtime.GOOS == "darwin" {
		return uint64(rusage.Maxrss)
	}
	return uint64(rusage.Maxrss) * 1024
}

func gpuGetCPUTimeNS() int64 {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0
	}
	return usage.Utime.Nano() + usage.Stime.Nano()
}

func gpuCountNaNInf(data []float32) (nanCount, infCount int) {
	for _, v := range data {
		if v != v {
			nanCount++
		} else if v > math.MaxFloat32 || v < -math.MaxFloat32 {
			infCount++
		}
	}
	return
}

func gpuMaxAbsF32(data []float32) float32 {
	m := float32(0)
	for _, v := range data {
		a := float32(math.Abs(float64(v)))
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) && a > m {
			m = a
		}
	}
	return m
}

func gpuParamInt(params map[string]any, key string) (int, bool) {
	v, ok := params[key]
	if !ok {
		return 0, false
	}
	n, ok := v.(int)
	return n, ok
}

type gpuTrialResult struct {
	timings       []int64
	cpuTimes      []int64
	warmupTimings []int64
	mem           benchMemoryInfo
	gc            benchGCInfo
	num           benchNumerical
}

func gpuRunTrials(warmup, trials int, setup func(), run func() []float32) gpuTrialResult {
	setup()

	warmupTimings := make([]int64, warmup)
	for i := 0; i < warmup; i++ {
		start := time.Now()
		run()
		warmupTimings[i] = time.Since(start).Nanoseconds()
	}

	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)
	var gcBefore debug.GCStats
	debug.ReadGCStats(&gcBefore)

	timings := make([]int64, trials)
	cpuTimes := make([]int64, trials)
	var lastOutput []float32
	for i := 0; i < trials; i++ {
		cpuBefore := gpuGetCPUTimeNS()
		start := time.Now()
		lastOutput = run()
		timings[i] = time.Since(start).Nanoseconds()
		cpuTimes[i] = gpuGetCPUTimeNS() - cpuBefore
	}

	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	var gcAfter debug.GCStats
	debug.ReadGCStats(&gcAfter)

	nanCount, infCount := 0, 0
	ma := float32(0)
	if lastOutput != nil {
		nanCount, infCount = gpuCountNaNInf(lastOutput)
		ma = gpuMaxAbsF32(lastOutput)
	}

	gcTimeNS := int64(gcAfter.PauseTotal - gcBefore.PauseTotal)

	return gpuTrialResult{
		timings:       timings,
		cpuTimes:      cpuTimes,
		warmupTimings: warmupTimings,
		mem: benchMemoryInfo{
			PeakRSSBytes: gpuGetPeakRSSBytes(),
			AllocBytes:   memAfter.TotalAlloc - memBefore.TotalAlloc,
		},
		gc: benchGCInfo{
			TotalGCTimeNS: gcTimeNS,
			GCPauseCount:  gcAfter.NumGC - gcBefore.NumGC,
		},
		num: benchNumerical{NaNCount: nanCount, InfCount: infCount, MaxAbs: ma},
	}
}

func gpuBuildScenario(id, axis string, params map[string]any, warmupCount, trialCount int, tr gpuTrialResult, knownFlops float64) benchScenario {
	sorted := make([]int64, len(tr.timings))
	for i, v := range tr.timings {
		sorted[i] = v
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	medNS := gpuMedianInt64(sorted)

	sortedCPU := make([]int64, len(tr.cpuTimes))
	for i, v := range tr.cpuTimes {
		sortedCPU[i] = v
	}
	sort.Slice(sortedCPU, func(i, j int) bool { return sortedCPU[i] < sortedCPU[j] })
	cpuMedian := gpuMedianInt64(sortedCPU)

	throughput := float64(0)
	batch, hasBatch := gpuParamInt(params, "batch")
	seqLen, hasSeq := gpuParamInt(params, "seq_len")
	if medNS > 0 && hasBatch && hasSeq {
		wallSec := float64(medNS) / 1e9
		throughput = float64(batch*seqLen) / wallSec
	}

	var derived benchDerived
	if medNS > 0 {
		medSec := float64(medNS) / 1e9
		derived.AllocRateBytesPerSec = float64(tr.mem.AllocBytes) / float64(trialCount) / medSec

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
		P95NS:                  gpuPercentileInt64(sorted, 95.0),
		MinNS:                  minNS,
		MaxNS:                  maxNS,
		IqrNS:                  gpuIqrInt64(sorted),
		CPUMedianNS:            cpuMedian,
		ThroughputTokensPerSec: throughput,
		Memory:                 tr.mem,
		GC:                     tr.gc,
		Numerical:              tr.num,
		Derived:                derived,
	}
}

// makeRandomModelSpec creates a ModelSpec with random weights for benchmarking.
func makeRandomModelSpec(rng *rand.Rand, vocabSize, hiddenDim, ffnDim, nLayers, nHeads, nKVHeads, headDim, nExperts, topK int) *ModelSpec {
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64()) * 0.02
		}
		return s
	}
	onesSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 1.0
		}
		return s
	}

	qDim := nHeads * headDim
	kvDim := nKVHeads * headDim

	blocks := make([]BlockSpec, nLayers)
	for i := range blocks {
		experts := make([]ExpertSpec, nExperts)
		for j := range experts {
			experts[j] = ExpertSpec{
				WGateWeight: randSlice(ffnDim * hiddenDim),
				WGateIn:     hiddenDim,
				WGateOut:    ffnDim,
				WUpWeight:   randSlice(ffnDim * hiddenDim),
				WUpIn:       hiddenDim,
				WUpOut:      ffnDim,
				WDownWeight: randSlice(hiddenDim * ffnDim),
				WDownIn:     ffnDim,
				WDownOut:    hiddenDim,
			}
		}
		blocks[i] = BlockSpec{
			AttnNormWeight:   onesSlice(hiddenDim),
			AttnNormEps:      1e-6,
			FfnNormWeight:    onesSlice(hiddenDim),
			FfnNormEps:       1e-6,
			WQWeight:         randSlice(qDim * hiddenDim),
			WQIn:             hiddenDim,
			WQOut:            qDim,
			WKWeight:         randSlice(kvDim * hiddenDim),
			WKIn:             hiddenDim,
			WKOut:            kvDim,
			WVWeight:         randSlice(kvDim * hiddenDim),
			WVIn:             hiddenDim,
			WVOut:            kvDim,
			WOWeight:         randSlice(hiddenDim * qDim),
			WOIn:             qDim,
			WOOut:            hiddenDim,
			RouterGateWeight: randSlice(nExperts * hiddenDim),
			RouterGateIn:     hiddenDim,
			RouterGateOut:    nExperts,
			Experts:          experts,
		}
	}

	return &ModelSpec{
		VocabSize:       vocabSize,
		HiddenDim:       hiddenDim,
		FFNDim:          ffnDim,
		NLayers:         nLayers,
		NHeads:          nHeads,
		NKVHeads:        nKVHeads,
		HeadDim:         headDim,
		NExperts:        nExperts,
		TopK:            topK,
		EmbeddingWeight: randSlice(vocabSize * hiddenDim),
		FinalNormWeight: onesSlice(hiddenDim),
		FinalNormEps:    1e-6,
		LmHeadWeight:    randSlice(vocabSize * hiddenDim),
		Blocks:          blocks,
	}
}

// --- TestBenchGPU: main entry point ---

func TestBenchGPU(t *testing.T) {
	if !MetalAvailable() {
		t.Skip("Metal not available on this platform")
	}

	ctx := NewMetalContext()
	if ctx == nil {
		t.Fatal("Failed to create MetalContext")
	}
	defer ctx.Close()
	if err := ctx.LoadRequiredShaders(benchGPUShaderDir()); err != nil {
		t.Fatalf("failed to load GPU shaders: %v", err)
	}

	result := benchResult{
		Metadata: benchMetadata{
			Language:        "go",
			Backend:         "gpu_metal",
			LanguageVersion: runtime.Version(),
			OS:              runtime.GOOS,
			CPUModel:        benchGpuCPUModel(),
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
			NTrials:         benchNTrials,
			NWarmup:         benchNWarmup,
			Seed:            benchSeed,
		},
	}

	// =====================================================================
	// Axis 6: GPU (axis="gpu")
	// =====================================================================

	// gpu_kernel_matmul: MPS GEMM 256x256
	{
		m, n, k := 256, 256, 256
		dataA := make([]float32, m*k)
		dataB := make([]float32, k*n)
		for i := range dataA {
			dataA[i] = float32(i%100) * 0.01
		}
		for i := range dataB {
			dataB[i] = float32(i%100) * 0.01
		}
		a := ctx.NewTensor(dataA, m, k)
		b := ctx.NewTensor(dataB, k, n)
		c := ctx.NewTensorZeros(m, n)
		defer a.Release()
		defer b.Release()
		defer c.Release()
		tr := gpuRunTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			ctx.Matmul(a, b, c, m, n, k)
			return nil
		})
		result.Scenarios = append(result.Scenarios, gpuBuildScenario("gpu_kernel_matmul", "gpu",
			map[string]any{"m": m, "n": n, "k": k},
			benchNWarmup, benchNTrials, tr, float64(2*m*n*k)))
	}

	// gpu_kernel_softmax: row-wise softmax on GPU
	{
		n := 1000
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i) * 0.001
		}
		input := ctx.NewTensor(data, 1, n)
		defer input.Release()
		tr := gpuRunTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := gpuSoftmaxRows(ctx, input, 1, n)
			out.Release()
			return nil
		})
		result.Scenarios = append(result.Scenarios, gpuBuildScenario("gpu_kernel_softmax", "gpu",
			map[string]any{"n": n},
			benchNWarmup, benchNTrials, tr, 4000))
	}

	// gpu_kernel_rmsnorm: RMSNorm on GPU
	{
		rows, hidden := 2, 64
		data := make([]float32, rows*hidden)
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		input := ctx.NewTensor(data, rows, hidden)
		weightVals := make([]float32, hidden)
		for i := range weightVals {
			weightVals[i] = 1.0
		}
		weight := ctx.NewTensor(weightVals, hidden)
		defer input.Release()
		defer weight.Release()
		tr := gpuRunTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			out := gpuRMSNorm(ctx, input, weight, rows, hidden, 1e-6)
			out.Release()
			return nil
		})
		result.Scenarios = append(result.Scenarios, gpuBuildScenario("gpu_kernel_rmsnorm", "gpu",
			map[string]any{"rows": rows, "hidden_dim": hidden},
			benchNWarmup, benchNTrials, tr, 12288))
	}

	rng := rand.New(rand.NewSource(benchSeed))

	// gpu_forward_{64,256,512}: full GPU forward pass (no CE loss)
	for _, sc := range []struct {
		label    string
		hidden   int
		ffnDim   int
		nLayers  int
		nHeads   int
		nKVHeads int
		headDim  int
		nExperts int
		topK     int
	}{
		{"64", 64, 128, 2, 2, 1, 32, 4, 2},
		{"256", 256, 512, 2, 4, 2, 64, 4, 2},
		{"512", 512, 1024, 2, 8, 2, 64, 4, 2},
	} {
		batch, seq := 2, 32
		vocabSize := 1000
		spec := makeRandomModelSpec(rng, vocabSize, sc.hidden, sc.ffnDim, sc.nLayers,
			sc.nHeads, sc.nKVHeads, sc.headDim, sc.nExperts, sc.topK)
		gpuModel := ctx.UploadModel(spec)
		inputIDs := make([]float32, batch*seq)
		for i := range inputIDs {
			inputIDs[i] = float32(i % vocabSize)
		}
		input := ctx.NewTensor(inputIDs, batch, seq)

		tr := gpuRunTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			logits := GpuForwardTensor(ctx, gpuModel, input, batch, seq)
			logits.Release()
			return nil
		})
		result.Scenarios = append(result.Scenarios, gpuBuildScenario(
			fmt.Sprintf("gpu_forward_%s", sc.label), "gpu",
			map[string]any{"batch": batch, "seq_len": seq, "hidden_dim": sc.hidden},
			benchNWarmup, benchNTrials, tr, 0))
		input.Release()
		gpuModel.ReleaseModel()
	}

	// gpu_train_{64,256,512}: GPU forward + GPU cross-entropy loss
	for _, sc := range []struct {
		label    string
		hidden   int
		ffnDim   int
		nLayers  int
		nHeads   int
		nKVHeads int
		headDim  int
		nExperts int
		topK     int
	}{
		{"64", 64, 128, 2, 2, 1, 32, 4, 2},
		{"256", 256, 512, 2, 4, 2, 64, 4, 2},
		{"512", 512, 1024, 2, 8, 2, 64, 4, 2},
	} {
		batch, seq := 2, 8
		vocabSize := 1000

		spec := makeRandomModelSpec(rng, vocabSize, sc.hidden, sc.ffnDim, sc.nLayers,
			sc.nHeads, sc.nKVHeads, sc.headDim, sc.nExperts, sc.topK)

		gpuModel := ctx.UploadModel(spec)

		inputIDs := make([]float32, batch*seq)
		targetIDs := make([]float32, batch*seq)
		for i := range inputIDs {
			inputIDs[i] = float32(i % vocabSize)
			targetIDs[i] = float32((i + 1) % vocabSize)
		}
		input := ctx.NewTensor(inputIDs, batch, seq)
		targets := ctx.NewTensor(targetIDs, batch*seq)

		tr := gpuRunTrials(benchNWarmup, benchNTrials, func() {}, func() []float32 {
			logits := GpuForwardTensor(ctx, gpuModel, input, batch, seq)
			lossSum := gpuCrossEntropyLossSum(ctx, logits, targets, batch*seq, vocabSize)
			logits.Release()
			lossSum.Release()
			return nil
		})
		result.Scenarios = append(result.Scenarios, gpuBuildScenario(
			fmt.Sprintf("gpu_train_%s", sc.label), "gpu",
			map[string]any{"batch": batch, "seq_len": seq, "hidden_dim": sc.hidden},
			benchNWarmup, benchNTrials, tr, 0))

		input.Release()
		targets.Release()
		gpuModel.ReleaseModel()
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
