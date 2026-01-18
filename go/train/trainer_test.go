package train

import (
	"testing"

	"github.com/fumi-engineer/machine_learning/go/model"
	"github.com/fumi-engineer/machine_learning/go/tensor"
)

func TestTrainConfig(t *testing.T) {
	cfg := DefaultTrainConfig()
	if cfg.LR != 1e-4 {
		t.Errorf("expected 1e-4, got %f", cfg.LR)
	}
	if cfg.Beta1 != 0.9 {
		t.Errorf("expected 0.9, got %f", cfg.Beta1)
	}
	if cfg.Beta2 != 0.95 {
		t.Errorf("expected 0.95, got %f", cfg.Beta2)
	}
}

func TestTrainerCreation(t *testing.T) {
	m := model.NewTiny()
	cfg := DefaultTrainConfig()
	trainer := NewTrainer(m, cfg)

	if trainer.Step() != 0 {
		t.Errorf("expected step 0, got %d", trainer.Step())
	}
}

func TestTrainerLRSchedule(t *testing.T) {
	m := model.NewTiny()
	cfg := DefaultTrainConfig()
	cfg.WarmupSteps = 100
	cfg.TotalSteps = 1000
	trainer := NewTrainer(m, cfg)

	// At step 0, LR should be 0 (warmup)
	if trainer.GetLR() != 0 {
		t.Errorf("expected LR 0 at step 0, got %f", trainer.GetLR())
	}

	// Simulate some steps
	for i := 0; i < 50; i++ {
		trainer.step++
	}

	// At step 50, should be halfway through warmup
	lr := trainer.GetLR()
	expected := cfg.LR * 0.5
	if lr < expected*0.9 || lr > expected*1.1 {
		t.Errorf("expected LR ~%f, got %f", expected, lr)
	}

	// After warmup, LR should decrease with cosine decay
	trainer.step = 100
	lrAtWarmup := trainer.GetLR()

	trainer.step = 500
	lrMid := trainer.GetLR()

	if lrMid >= lrAtWarmup {
		t.Errorf("LR should decrease after warmup: %f >= %f", lrMid, lrAtWarmup)
	}
}

func TestTrainStep(t *testing.T) {
	m := model.NewTiny()
	cfg := DefaultTrainConfig()
	trainer := NewTrainer(m, cfg)

	// Create input [batch=2, seq_len=8]
	batch := 2
	seqLen := 8
	inputData := make([]float32, batch*seqLen)
	targetData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % 100) // Valid token IDs 0-99
		targetData[i] = float32((i + 1) % 100)
	}

	input := tensor.FromSlice(inputData, tensor.NewShape(batch, seqLen))
	targets := tensor.FromSlice(targetData, tensor.NewShape(batch, seqLen))

	loss := trainer.TrainStep(input, targets)

	if loss < 0 {
		t.Errorf("expected non-negative loss, got %f", loss)
	}

	if trainer.Step() != 1 {
		t.Errorf("expected step 1, got %d", trainer.Step())
	}
}

func TestMultipleTrainSteps(t *testing.T) {
	m := model.NewTiny()
	cfg := DefaultTrainConfig()
	trainer := NewTrainer(m, cfg)

	// Create input
	batch := 1
	seqLen := 4
	inputData := make([]float32, batch*seqLen)
	targetData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i % 100)
		targetData[i] = float32((i + 1) % 100)
	}

	input := tensor.FromSlice(inputData, tensor.NewShape(batch, seqLen))
	targets := tensor.FromSlice(targetData, tensor.NewShape(batch, seqLen))

	// Run multiple steps
	var losses []float32
	for i := 0; i < 5; i++ {
		loss := trainer.TrainStep(input, targets)
		losses = append(losses, loss)
	}

	if trainer.Step() != 5 {
		t.Errorf("expected step 5, got %d", trainer.Step())
	}

	// All losses should be valid
	for i, loss := range losses {
		if loss < 0 {
			t.Errorf("step %d: expected non-negative loss, got %f", i, loss)
		}
	}
}
