package model

import (
	"testing"

	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

func TestConfig(t *testing.T) {
	cfg := Default6_9B()
	if cfg.HiddenDim != 768 {
		t.Errorf("expected 768, got %d", cfg.HiddenDim)
	}
	if cfg.NLayers != 30 {
		t.Errorf("expected 30, got %d", cfg.NLayers)
	}
	if cfg.NExperts != 16 {
		t.Errorf("expected 16, got %d", cfg.NExperts)
	}
	if cfg.TopKExperts != 4 {
		t.Errorf("expected 4, got %d", cfg.TopKExperts)
	}
}

func TestTinyConfig(t *testing.T) {
	cfg := Tiny()
	if cfg.HiddenDim != 64 {
		t.Errorf("expected 64, got %d", cfg.HiddenDim)
	}
	if cfg.NLayers != 2 {
		t.Errorf("expected 2, got %d", cfg.NLayers)
	}
}

func TestConfigParams(t *testing.T) {
	cfg := Default6_9B()
	total := cfg.TotalParams()
	active := cfg.ActiveParams()

	// Rough checks
	if total < 6_000_000_000 || total > 8_000_000_000 {
		t.Errorf("unexpected total params: %d", total)
	}
	if active < 1_500_000_000 || active > 2_500_000_000 {
		t.Errorf("unexpected active params: %d", active)
	}
	if active >= total {
		t.Errorf("active should be less than total")
	}
}

func TestModelCreation(t *testing.T) {
	model := NewTiny()
	if model.Config().HiddenDim != 64 {
		t.Errorf("expected 64, got %d", model.Config().HiddenDim)
	}
	if model.NumLayers() != 2 {
		t.Errorf("expected 2, got %d", model.NumLayers())
	}
}

func TestModelForward(t *testing.T) {
	model := NewTiny()

	// Create input [batch=1, seq_len=4]
	tokenIDs := []int{10, 20, 30, 40}
	logits := model.ForwardIDs(tokenIDs, 1, 4)

	// Output should be [1, 4, vocab_size=1000]
	expected := tensor.NewShape(1, 4, 1000)
	if !logits.Shape().Equal(expected) {
		t.Errorf("expected shape %v, got %v", expected, logits.Shape())
	}
}

func TestModelBackward(t *testing.T) {
	model := NewTiny()

	// Forward pass
	tokenIDs := []int{10, 20, 30, 40}
	logits := model.ForwardIDs(tokenIDs, 1, 4)

	// Backward pass
	gradOutput := tensor.Ones(logits.Shape(), tensor.F32)
	gradInput := model.Backward(gradOutput)

	// Should return gradient w.r.t. hidden states
	if gradInput == nil {
		t.Error("expected non-nil gradient")
	}
}

func TestModelParameters(t *testing.T) {
	model := NewTiny()
	params := model.Parameters()
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}

func TestRouter(t *testing.T) {
	router := NewRouter(64, 4, 2)

	// Input [batch=1, seq_len=2, hidden_dim=64]
	input := tensor.Randn(tensor.NewShape(1, 2, 64), tensor.F32)
	weights, indices := router.Forward(input)

	// weights should be [2, 2] (2 tokens, top-2)
	if !weights.Shape().Equal(tensor.NewShape(2, 2)) {
		t.Errorf("expected shape [2,2], got %v", weights.Shape())
	}

	// indices should have 2 tokens
	if len(indices) != 2 {
		t.Errorf("expected 2 index sets, got %d", len(indices))
	}

	// Each token should have top-2 indices
	for i, idx := range indices {
		if len(idx) != 2 {
			t.Errorf("token %d: expected 2 indices, got %d", i, len(idx))
		}
	}

	// Weights should sum to 1 per token
	weightsData := weights.DataPtr()
	for i := 0; i < 2; i++ {
		sum := weightsData[i*2] + weightsData[i*2+1]
		if sum < 0.99 || sum > 1.01 {
			t.Errorf("token %d: weights sum to %f, expected ~1.0", i, sum)
		}
	}
}

func TestMoELayer(t *testing.T) {
	moe := NewMoELayer(64, 256, 4, 2)

	// Input [batch=1, seq_len=2, hidden_dim=64]
	input := tensor.Randn(tensor.NewShape(1, 2, 64), tensor.F32)
	output := moe.Forward(input)

	// Output should be same shape as input
	if !output.Shape().Equal(input.Shape()) {
		t.Errorf("expected shape %v, got %v", input.Shape(), output.Shape())
	}
}

func TestAuxLoss(t *testing.T) {
	router := NewRouter(64, 4, 2)

	// Forward to compute aux loss
	input := tensor.Randn(tensor.NewShape(1, 8, 64), tensor.F32)
	router.Forward(input)

	auxLoss := router.ComputeAuxLoss(0.01)
	if auxLoss < 0 {
		t.Error("aux loss should be non-negative")
	}
}

func TestTransformerBlock(t *testing.T) {
	cfg := Tiny()
	block := NewTransformerBlock(cfg)

	// Input [batch=1, seq_len=4, hidden_dim=64]
	input := tensor.Randn(tensor.NewShape(1, 4, 64), tensor.F32)
	output := block.Forward(input)

	// Output should be same shape
	if !output.Shape().Equal(input.Shape()) {
		t.Errorf("expected shape %v, got %v", input.Shape(), output.Shape())
	}

	// Block should have parameters
	params := block.Parameters()
	if len(params) == 0 {
		t.Error("expected non-empty parameters")
	}
}
