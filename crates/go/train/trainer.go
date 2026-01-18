// Package train provides training utilities for the MoE Transformer.
package train

import (
	"math"

	"github.com/pikafumi/machine_learning/crates/go/model"
	"github.com/pikafumi/machine_learning/crates/go/tensor"
)

// TrainConfig holds training configuration.
type TrainConfig struct {
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32
	GradClip    float32
	WarmupSteps int
	TotalSteps  int
	AuxAlpha    float32
}

// DefaultTrainConfig returns default training configuration.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		LR:          1e-4,
		Beta1:       0.9,
		Beta2:       0.95,
		Eps:         1e-8,
		WeightDecay: 0.1,
		GradClip:    1.0,
		WarmupSteps: 1000,
		TotalSteps:  100000,
		AuxAlpha:    0.01,
	}
}

// AdamWState holds optimizer state for a parameter.
type AdamWState struct {
	M *tensor.Tensor // First moment
	V *tensor.Tensor // Second moment
}

// Trainer handles model training.
type Trainer struct {
	model      *model.MoETransformer
	config     TrainConfig
	step       int
	states     []AdamWState
}

// NewTrainer creates a new trainer.
func NewTrainer(m *model.MoETransformer, cfg TrainConfig) *Trainer {
	params := m.Parameters()
	states := make([]AdamWState, len(params))
	for i, p := range params {
		states[i] = AdamWState{
			M: tensor.Zeros(p.Shape(), tensor.F32),
			V: tensor.Zeros(p.Shape(), tensor.F32),
		}
	}

	return &Trainer{
		model:  m,
		config: cfg,
		step:   0,
		states: states,
	}
}

// GetLR returns the current learning rate with warmup and cosine decay.
func (t *Trainer) GetLR() float32 {
	if t.step < t.config.WarmupSteps {
		// Linear warmup
		return t.config.LR * float32(t.step) / float32(t.config.WarmupSteps)
	}

	// Cosine decay
	progress := float32(t.step-t.config.WarmupSteps) / float32(t.config.TotalSteps-t.config.WarmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}

	minLR := t.config.LR * 0.1
	return minLR + 0.5*(t.config.LR-minLR)*(1.0+float32(math.Cos(math.Pi*float64(progress))))
}

// Step returns the current step.
func (t *Trainer) Step() int {
	return t.step
}

// crossEntropyLoss computes cross entropy loss.
func crossEntropyLoss(logits, targets *tensor.Tensor) float32 {
	dims := logits.Shape().Dims()
	batch := dims[0]
	seqLen := dims[1]
	vocabSize := dims[2]

	logitsData := logits.DataPtr()
	targetsData := targets.DataPtr()

	totalLoss := float32(0.0)
	numTokens := batch * seqLen

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			offset := (b*seqLen + s) * vocabSize
			targetIdx := int(targetsData[b*seqLen+s])

			// Compute log softmax
			maxVal := logitsData[offset]
			for v := 1; v < vocabSize; v++ {
				if logitsData[offset+v] > maxVal {
					maxVal = logitsData[offset+v]
				}
			}

			sumExp := float32(0.0)
			for v := 0; v < vocabSize; v++ {
				sumExp += float32(math.Exp(float64(logitsData[offset+v] - maxVal)))
			}

			logProb := logitsData[offset+targetIdx] - maxVal - float32(math.Log(float64(sumExp)))
			totalLoss -= logProb
		}
	}

	return totalLoss / float32(numTokens)
}

// TrainStep performs one training step.
func (t *Trainer) TrainStep(input, targets *tensor.Tensor) float32 {
	t.step++

	// Forward pass
	logits := t.model.Forward(input)

	// Compute loss
	loss := crossEntropyLoss(logits, targets)

	// Add auxiliary loss
	auxLoss := t.model.TotalAuxLoss(t.config.AuxAlpha)
	totalLoss := loss + auxLoss

	// Backward pass (simplified - just creates dummy gradients)
	gradOutput := tensor.Ones(logits.Shape(), tensor.F32)
	_ = t.model.Backward(gradOutput)

	// Get current LR
	lr := t.GetLR()

	// AdamW update
	params := t.model.Parameters()
	for i, param := range params {
		paramData := param.DataPtr()
		mData := t.states[i].M.DataPtr()
		vData := t.states[i].V.DataPtr()

		// Dummy gradient (in real impl, would use computed gradients)
		for j := range paramData {
			grad := float32(0.001) // Placeholder

			// Update moments
			mData[j] = t.config.Beta1*mData[j] + (1-t.config.Beta1)*grad
			vData[j] = t.config.Beta2*vData[j] + (1-t.config.Beta2)*grad*grad

			// Bias correction
			mHat := mData[j] / (1 - float32(math.Pow(float64(t.config.Beta1), float64(t.step))))
			vHat := vData[j] / (1 - float32(math.Pow(float64(t.config.Beta2), float64(t.step))))

			// Update with weight decay
			paramData[j] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+t.config.Eps) + t.config.WeightDecay*paramData[j])
		}
	}

	return totalLoss
}
