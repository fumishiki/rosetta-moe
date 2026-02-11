// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

// Tests for the MoE Transformer implementation.
//
// Testing philosophy: test module boundaries and exported behavior, not internals.
// The type system enforces most invariants (shapes, dtypes); tests focus on
// cross-layer integration, numerical correctness at seams, and training convergence.

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Cross-module seam: Tensor -> Linear.
// Verifies that Linear correctly performs y = x @ W^T with known weights.
func TestTensorLinearSeamForward(t *testing.T) {
	input := FromSlice([]float32{1, 2, 3, 4}, NewShape(2, 2))
	layer := NewLinear(2, 3, false)

	// Override weights with a known matrix for deterministic testing.
	// W = [[1,0],[0,1],[1,1]], so y = x @ W^T = [[1,2,3],[3,4,7]]
	copy(layer.weight.DataPtr(), []float32{
		1, 0,
		0, 1,
		1, 1,
	})

	output := layer.Forward(input)
	if !output.Shape().Equal(NewShape(2, 3)) {
		t.Fatalf("expected shape [2, 3], got %v", output.Shape())
	}

	got := output.DataPtr()
	want := []float32{1, 2, 3, 3, 4, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: expected %f, got %f", i, want[i], got[i])
		}
	}
}

// End-to-end forward pass: token IDs -> logits with correct output shape.
func TestModelForward(t *testing.T) {
	m := NewTiny()
	tokenIDs := []int{10, 20, 30, 40}
	logits := m.ForwardIDs(tokenIDs, 1, 4)

	expected := NewShape(1, 4, 1000)
	if !logits.Shape().Equal(expected) {
		t.Errorf("expected shape %v, got %v", expected, logits.Shape())
	}
}

// Verify backward pass produces non-nil gradients.
func TestModelBackward(t *testing.T) {
	m := NewTiny()
	tokenIDs := []int{10, 20, 30, 40}
	logits := m.ForwardIDs(tokenIDs, 1, 4)

	gradOutput := Ones(logits.Shape(), F32)
	gradInput := m.Backward(gradOutput)

	if gradInput == nil {
		t.Error("expected non-nil gradient")
	}
}

// Verify Generate with explicit strategy matches GenerateGreedy.
func TestGenerateWithStrategy(t *testing.T) {
	m := NewTiny()
	prompt := []int{1, 2, 3}

	result := Generate(m, prompt, 8, GreedySampling{})
	if len(result) != 8 {
		t.Fatalf("expected length 8, got %d", len(result))
	}
	if result[0] != 1 || result[1] != 2 || result[2] != 3 {
		t.Fatalf("expected prompt preserved, got %v", result[:3])
	}

	greedy := m.GenerateGreedy(prompt, 8)
	for i := range result {
		if result[i] != greedy[i] {
			t.Fatalf("Generate(GreedySampling) != GenerateGreedy at index %d: %d vs %d", i, result[i], greedy[i])
		}
	}
}

// Verify all four sampling strategies produce correct-length output.
func TestGenerationInterfaces(t *testing.T) {
	m := NewTiny()
	prompt := []int{1, 2, 3}

	greedy := m.GenerateGreedy(prompt, 8)
	if len(greedy) != 8 {
		t.Fatalf("expected greedy length 8, got %d", len(greedy))
	}

	sampled := m.GenerateSample(prompt, 8, 1.0, 42)
	if len(sampled) != 8 {
		t.Fatalf("expected sampled length 8, got %d", len(sampled))
	}

	topk := m.GenerateTopK(prompt, 8, 10, 1.0, 42)
	if len(topk) != 8 {
		t.Fatalf("expected top-k length 8, got %d", len(topk))
	}

	topp := m.GenerateTopP(prompt, 8, 0.9, 1.0, 42)
	if len(topp) != 8 {
		t.Fatalf("expected top-p length 8, got %d", len(topp))
	}
}

// Router: verify shape, index count, and weight normalization (sum ~= 1.0).
func TestRouter(t *testing.T) {
	router := NewRouter(64, 4, 2)

	input := Randn(NewShape(1, 2, 64), F32)
	weights, indices := router.Forward(input)

	if !weights.Shape().Equal(NewShape(2, 2)) {
		t.Errorf("expected shape [2,2], got %v", weights.Shape())
	}

	if len(indices) != 2 {
		t.Errorf("expected 2 index sets, got %d", len(indices))
	}

	for i, idx := range indices {
		if len(idx) != 2 {
			t.Errorf("token %d: expected 2 indices, got %d", i, len(idx))
		}
	}

	weightsData := weights.DataPtr()
	for i := 0; i < 2; i++ {
		sum := weightsData[i*2] + weightsData[i*2+1]
		if sum < 0.99 || sum > 1.01 {
			t.Errorf("token %d: weights sum to %f, expected ~1.0", i, sum)
		}
	}
}

// MoE layer: output shape must match input shape.
func TestMoELayer(t *testing.T) {
	moe := NewMoELayer(64, 256, 4, 2)

	input := Randn(NewShape(1, 2, 64), F32)
	output := moe.Forward(input)

	if !output.Shape().Equal(input.Shape()) {
		t.Errorf("expected shape %v, got %v", input.Shape(), output.Shape())
	}
}

// Auxiliary loss should be non-negative.
func TestAuxLoss(t *testing.T) {
	router := NewRouter(64, 4, 2)

	input := Randn(NewShape(1, 8, 64), F32)
	router.Forward(input)

	auxLoss := router.ComputeAuxLoss(0.01)
	if auxLoss < 0 {
		t.Error("aux loss should be non-negative")
	}
}

// TransformerBlock: shape preservation and non-empty parameters.
func TestTransformerBlock(t *testing.T) {
	cfg := Tiny()
	block := NewTransformerBlock(cfg)

	input := Randn(NewShape(1, 4, 64), F32)
	output := block.Forward(input)

	if !output.Shape().Equal(input.Shape()) {
		t.Errorf("expected shape %v, got %v", input.Shape(), output.Shape())
	}

	params := block.Parameters()
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}

// Single training step: loss should be non-negative, step counter should advance.
func TestTrainStep(t *testing.T) {
	m := NewTiny()
	cfg := DefaultTrainConfig()
	trainer := NewTrainer(m, cfg)

	batch := 2
	seqLen := 8
	inputData := make([]float32, batch*seqLen)
	targetData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % 100)
		targetData[i] = float32((i + 1) % 100)
	}

	input := FromSlice(inputData, NewShape(batch, seqLen))
	targets := FromSlice(targetData, NewShape(batch, seqLen))

	loss := trainer.TrainStep(input, targets)

	if loss < 0 {
		t.Errorf("expected non-negative loss, got %f", loss)
	}
	if trainer.Step() != 1 {
		t.Errorf("expected step 1, got %d", trainer.Step())
	}
}

// Cross-entropy gradient: each row should sum to ~0 (softmax gradient property).
func TestCrossEntropyGradShape(t *testing.T) {
	logits := FromSlice(
		[]float32{
			1, 2, 3,
			2, 1, 0,
		},
		NewShape(1, 2, 3),
	)
	targets := FromSlice([]float32{2, 0}, NewShape(1, 2))
	grad := crossEntropyGrad(logits, targets)

	if !grad.Shape().Equal(logits.Shape()) {
		t.Fatalf("expected grad shape %v, got %v", logits.Shape(), grad.Shape())
	}

	// Each row of softmax gradient sums to 0: sum(softmax) = 1, minus 1 for target.
	row0 := grad.DataPtr()[:3]
	row1 := grad.DataPtr()[3:6]
	sum0 := row0[0] + row0[1] + row0[2]
	sum1 := row1[0] + row1[1] + row1[2]
	if math.Abs(float64(sum0)) > 1e-4 || math.Abs(float64(sum1)) > 1e-4 {
		t.Fatalf("expected per-row grad sums ~0, got %f and %f", sum0, sum1)
	}
}

// Multiple training steps: all losses should be non-negative.
func TestMultipleTrainSteps(t *testing.T) {
	m := NewTiny()
	cfg := DefaultTrainConfig()
	trainer := NewTrainer(m, cfg)

	batch := 1
	seqLen := 4
	inputData := make([]float32, batch*seqLen)
	targetData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % 100)
		targetData[i] = float32((i + 1) % 100)
	}

	input := FromSlice(inputData, NewShape(batch, seqLen))
	targets := FromSlice(targetData, NewShape(batch, seqLen))

	var losses []float32
	for i := 0; i < 5; i++ {
		loss := trainer.TrainStep(input, targets)
		losses = append(losses, loss)
	}

	if trainer.Step() != 5 {
		t.Errorf("expected step 5, got %d", trainer.Step())
	}

	for i, loss := range losses {
		if loss < 0 {
			t.Errorf("step %d: expected non-negative loss, got %f", i, loss)
		}
	}
}

// --- Tensor and Shape unit tests ---

func TestShape(t *testing.T) {
	s := NewShape(2, 3, 4)
	if s.NDim() != 3 {
		t.Errorf("expected 3 dims, got %d", s.NDim())
	}
	if s.Numel() != 24 {
		t.Errorf("expected 24 elements, got %d", s.Numel())
	}
	if s.At(0) != 2 || s.At(1) != 3 || s.At(2) != 4 {
		t.Errorf("unexpected dims: %v", s.Dims())
	}
}

func TestShapeStrides(t *testing.T) {
	s := NewShape(2, 3, 4)
	strides := s.Strides()
	if len(strides) != 3 {
		t.Fatalf("expected 3 strides, got %d", len(strides))
	}
	// Row-major: [12, 4, 1]
	if strides[0] != 12 || strides[1] != 4 || strides[2] != 1 {
		t.Errorf("unexpected strides: %v", strides)
	}
}

func TestTensorZeros(t *testing.T) {
	tensor := Zeros(NewShape(2, 3), F32)
	if tensor.Shape().Numel() != 6 {
		t.Errorf("expected 6 elements, got %d", tensor.Shape().Numel())
	}
	for _, v := range tensor.Data() {
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	}
}

func TestTensorOnes(t *testing.T) {
	tensor := Ones(NewShape(2, 3), F32)
	for _, v := range tensor.Data() {
		if v != 1 {
			t.Errorf("expected 1, got %f", v)
		}
	}
}

func TestTensorFromSlice(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor := FromSlice(data, NewShape(2, 3))
	if tensor.At(0, 0) != 1 || tensor.At(1, 2) != 6 {
		t.Errorf("unexpected values")
	}
}

func TestTensorAdd(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	b := FromSlice([]float32{4, 5, 6}, NewShape(3))
	c := a.Add(b)
	data := c.Data()
	if data[0] != 5 || data[1] != 7 || data[2] != 9 {
		t.Errorf("unexpected sum: %v", data)
	}
}

func TestTensorMul(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	b := FromSlice([]float32{4, 5, 6}, NewShape(3))
	c := a.Mul(b)
	data := c.Data()
	if data[0] != 4 || data[1] != 10 || data[2] != 18 {
		t.Errorf("unexpected product: %v", data)
	}
}

func TestTensorScale(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	c := a.Scale(2)
	data := c.Data()
	if data[0] != 2 || data[1] != 4 || data[2] != 6 {
		t.Errorf("unexpected scaled: %v", data)
	}
}

func TestTensorSiLU(t *testing.T) {
	a := FromSlice([]float32{0, 1, -1}, NewShape(3))
	c := a.SiLU()
	data := c.Data()
	// SiLU(0) = 0, SiLU(1) ~ 0.731, SiLU(-1) ~ -0.269
	if math.Abs(float64(data[0])) > 0.001 {
		t.Errorf("expected ~0, got %f", data[0])
	}
	if math.Abs(float64(data[1])-0.731) > 0.01 {
		t.Errorf("expected ~0.731, got %f", data[1])
	}
}

func TestTensorSoftmax(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(1, 3))
	c := a.Softmax()
	data := c.Data()
	sum := data[0] + data[1] + data[2]
	if math.Abs(float64(sum)-1.0) > 0.001 {
		t.Errorf("expected sum 1, got %f", sum)
	}
	// Should be monotonically increasing
	if data[0] >= data[1] || data[1] >= data[2] {
		t.Errorf("expected monotonic increase: %v", data)
	}
}

func TestMatmul(t *testing.T) {
	// [2, 3] x [3, 4] -> [2, 4]
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, NewShape(2, 3))
	b := FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, NewShape(3, 4))
	c := Matmul(a, b)

	if !c.Shape().Equal(NewShape(2, 4)) {
		t.Errorf("unexpected shape: %v", c.Shape())
	}

	// c[0,0] = 1*1 + 2*5 + 3*9 = 38
	if c.At(0, 0) != 38 {
		t.Errorf("expected 38, got %f", c.At(0, 0))
	}
}

func TestTranspose(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, NewShape(2, 3))
	b := a.Transpose()
	if !b.Shape().Equal(NewShape(3, 2)) {
		t.Errorf("unexpected shape: %v", b.Shape())
	}
	if b.At(0, 0) != 1 || b.At(0, 1) != 4 || b.At(1, 0) != 2 {
		t.Errorf("unexpected values after transpose")
	}
}

func TestDType(t *testing.T) {
	if F32.Size() != 4 {
		t.Errorf("expected F32 size 4, got %d", F32.Size())
	}
	if F16.Size() != 2 {
		t.Errorf("expected F16 size 2, got %d", F16.Size())
	}
	if F32.String() != "f32" {
		t.Errorf("expected 'f32', got '%s'", F32.String())
	}
}

func TestBroadcast(t *testing.T) {
	a := NewShape(3, 1, 5)
	b := NewShape(4, 5)
	c, err := Broadcast(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !c.Equal(NewShape(3, 4, 5)) {
		t.Errorf("expected [3,4,5], got %v", c)
	}
}

func TestBroadcastError(t *testing.T) {
	a := NewShape(3, 4)
	b := NewShape(5, 4)
	_, err := Broadcast(a, b)
	if err == nil {
		t.Error("expected broadcast error")
	}
}

// --- Training infrastructure tests ---

func TestCheckpointStorage(t *testing.T) {
	store := NewCheckpointStorage(true)
	if !store.Enabled() {
		t.Error("expected enabled")
	}

	tensor := Ones(NewShape(4), F32)
	store.Save(0, tensor)

	if store.Get(0) == nil {
		t.Error("expected non-nil tensor at index 0")
	}
	if store.Len() != 1 {
		t.Errorf("expected len 1, got %d", store.Len())
	}

	store.Clear()
	if store.Len() != 0 {
		t.Errorf("expected len 0 after clear, got %d", store.Len())
	}
}

func TestCheckpointContext(t *testing.T) {
	ctx := NewCheckpointContext(2)

	if !ctx.ShouldCheckpoint(0) {
		t.Error("expected checkpoint at index 0")
	}
	if ctx.ShouldCheckpoint(1) {
		t.Error("did not expect checkpoint at index 1")
	}
	if !ctx.ShouldCheckpoint(2) {
		t.Error("expected checkpoint at index 2")
	}
	if ctx.ShouldCheckpoint(3) {
		t.Error("did not expect checkpoint at index 3")
	}
}

// LR schedule: warmup=0 -> lr=0, warmup=end -> lr=peak, past total -> lr >= min.
func TestLRSchedule(t *testing.T) {
	m := NewTiny()
	cfg := TrainConfig{
		LR:          1e-3,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.1,
		GradClip:    1.0,
		WarmupSteps: 100,
		TotalSteps:  1000,
		AuxAlpha:    0.01,
	}
	trainer := NewTrainer(m, cfg)

	// step 0 => lr == 0
	trainer.step = 0
	lr0 := trainer.GetLR()
	if lr0 != 0.0 {
		t.Errorf("expected lr 0 at step 0, got %f", lr0)
	}

	// at warmup_steps => lr == config.LR
	trainer.step = cfg.WarmupSteps
	lrWarmup := trainer.GetLR()
	diff := lrWarmup - cfg.LR
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-6 {
		t.Errorf("expected lr %f at warmup end, got %f", cfg.LR, lrWarmup)
	}

	// well past total_steps => lr >= min_lr
	trainer.step = cfg.TotalSteps * 10
	lrEnd := trainer.GetLR()
	minLR := cfg.LR * 0.1
	if lrEnd < minLR-1e-7 {
		t.Errorf("expected lr >= %f, got %f", minLR, lrEnd)
	}
}

// Gradient clipping: after clip, L2 norm should be <= clip_norm.
func TestGradClip(t *testing.T) {
	data := make([]float32, 10)
	for i := range data {
		data[i] = 100.0
	}
	tensor := FromSlice(data, NewShape(10))

	norm := clipTensorByGlobalNorm(tensor, 1.0)
	if norm <= 0 {
		t.Error("expected positive original norm")
	}

	clippedData := tensor.DataPtr()
	sumSq := float32(0)
	for _, v := range clippedData {
		sumSq += v * v
	}
	clippedNorm := SqrtF32(sumSq)
	if clippedNorm > 1.0+1e-4 {
		t.Errorf("expected clipped norm <= 1.0, got %f", clippedNorm)
	}
}

func TestLossScaler(t *testing.T) {
	scaler := DynamicLossScaler()

	if scaler.Scale() != 65536.0 {
		t.Errorf("expected init scale 65536, got %f", scaler.Scale())
	}

	scaled := scaler.ScaleLoss(1.0)
	if scaled != 65536.0 {
		t.Errorf("expected scaled loss 65536, got %f", scaled)
	}

	unscaled := scaler.UnscaleGrads(65536.0)
	if unscaled != 1.0 {
		t.Errorf("expected unscaled grad 1.0, got %f", unscaled)
	}

	if scaler.ShouldSkipStep() {
		t.Error("should not skip step initially")
	}
}

func TestMixedPrecisionConfig(t *testing.T) {
	cfg := DefaultMixedPrecisionConfig()
	if cfg.Enabled {
		t.Error("default should be disabled")
	}

	if !cfg.IsFP32Layer("final_norm") {
		t.Error("expected final_norm to be FP32")
	}
	if !cfg.IsFP32Layer("lm_head") {
		t.Error("expected lm_head to be FP32")
	}
	if cfg.IsFP32Layer("attention") {
		t.Error("did not expect attention to be FP32")
	}
}

func TestMasterWeights(t *testing.T) {
	params := []*Tensor{
		Ones(NewShape(4, 8), F32),
		Ones(NewShape(16), F32),
	}
	mw := NewMasterWeights(params)

	if mw.Len() != 2 {
		t.Errorf("expected 2 master weights, got %d", mw.Len())
	}

	weights := mw.Weights()
	if !weights[0].Shape().Equal(NewShape(4, 8)) {
		t.Errorf("expected shape [4,8], got %v", weights[0].Shape())
	}
}

// TestConvergence trains the tiny model for 200 steps and outputs loss convergence.
// This test verifies that the training loop reduces loss over multiple iterations,
// demonstrating that the optimizer and gradient computation are functioning correctly.
func TestConvergence(t *testing.T) {
	m := NewTiny()
	cfg := TrainConfig{
		LR:          1e-3,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.1,
		GradClip:    1.0,
		WarmupSteps: 10,
		TotalSteps:  1200,
		AuxAlpha:    0.01,
	}
	trainer := NewTrainer(m, cfg)

	batch, seqLen := 2, 8
	inputData := make([]float32, batch*seqLen)
	targetData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % 1000)
		targetData[i] = float32((i + 1) % 1000)
	}
	input := FromSlice(inputData, NewShape(batch, seqLen))
	targets := FromSlice(targetData, NewShape(batch, seqLen))

	nSteps := 1000
	losses := make([]float32, nSteps)
	for i := 0; i < nSteps; i++ {
		loss := trainer.TrainStep(input, targets)
		losses[i] = loss
	}

	// Print JSON
	lossStrs := make([]string, nSteps)
	for i, l := range losses {
		lossStrs[i] = fmt.Sprintf("%.6f", l)
	}
	fmt.Printf("{\"language\":\"go\",\"steps\":%d,\"losses\":[%s]}\n", nSteps, strings.Join(lossStrs, ","))

	// Check that average loss decreased from first quarter to last quarter
	// (more robust than single-point comparison, tolerates noise from aux loss)
	quarter := nSteps / 4
	firstQuarterAvg := float32(0)
	lastQuarterAvg := float32(0)
	for i := 0; i < quarter; i++ {
		firstQuarterAvg += losses[i]
		lastQuarterAvg += losses[nSteps-quarter+i]
	}
	firstQuarterAvg /= float32(quarter)
	lastQuarterAvg /= float32(quarter)

	if lastQuarterAvg >= firstQuarterAvg {
		t.Errorf("loss did not decrease: first_quarter_avg=%.6f last_quarter_avg=%.6f",
			firstQuarterAvg, lastQuarterAvg)
	}
}
