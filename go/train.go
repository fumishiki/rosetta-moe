// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package nn

import "strings"

// TrainConfig holds optimizer and training hyperparameters.
type TrainConfig struct {
	LR          float32 // peak learning rate
	Beta1       float32 // AdamW first moment decay
	Beta2       float32 // AdamW second moment decay
	Eps         float32 // AdamW epsilon (numerical stability)
	WeightDecay float32 // AdamW weight decay coefficient
	GradClip    float32 // max gradient L2 norm
	WarmupSteps int     // linear warmup phase length
	TotalSteps  int     // total training steps (for cosine schedule)
	AuxAlpha    float32 // MoE auxiliary loss coefficient
}

// DefaultTrainConfig returns standard training hyperparameters.
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

// AdamWState holds the first and second moment estimates for one parameter tensor.
type AdamWState struct {
	M *Tensor // first moment (mean of gradients)
	V *Tensor // second moment (mean of squared gradients)
}

// Trainer encapsulates the model, optimizer state, and LR schedule.
type Trainer struct {
	model  *MoETransformer
	config TrainConfig
	step   int
	states []AdamWState // one per parameter tensor
}

// NewTrainer creates a Trainer with AdamW optimizer state initialized to zero.
func NewTrainer(m *MoETransformer, cfg TrainConfig) *Trainer {
	params := m.Parameters()
	states := make([]AdamWState, len(params))
	for i, p := range params {
		states[i] = AdamWState{
			M: Zeros(p.Shape(), F32),
			V: Zeros(p.Shape(), F32),
		}
	}
	return &Trainer{model: m, config: cfg, states: states}
}

// GetLR computes the current learning rate using linear warmup + cosine decay.
//
//   warmup:  lr = peak_lr * step / warmup_steps
//   cosine:  lr = min_lr + 0.5*(peak_lr - min_lr)*(1 + cos(pi * progress))
//   min_lr = 0.1 * peak_lr
//
// This schedule ramps up linearly to prevent training instability at the start,
// then smoothly decays to 10% of peak.
func (t *Trainer) GetLR() float32 {
	if t.step < t.config.WarmupSteps {
		return t.config.LR * float32(t.step) / float32(t.config.WarmupSteps)
	}
	progress := float32(t.step-t.config.WarmupSteps) / float32(t.config.TotalSteps-t.config.WarmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	minLR := t.config.LR * 0.1
	return minLR + 0.5*(t.config.LR-minLR)*(1.0+CosF32(3.1415927*progress))
}

// Step returns the current training step count.
func (t *Trainer) Step() int { return t.step }

// crossEntropyLoss computes the mean cross-entropy loss over all positions.
//
//   L = -(1/N) * sum_{b,s} log(softmax(logits[b,s])[target[b,s]])
//
// Numerically stable via log-sum-exp:
//   log(softmax(x)_i) = x_i - max(x) - log(sum(exp(x - max(x))))
func crossEntropyLoss(logits, targets *Tensor) float32 {
	dims := logits.Shape().DimsRef()
	batch, seqLen, vocabSize := dims[0], dims[1], dims[2]

	logitsData := logits.DataPtr()
	targetsData := targets.DataPtr()

	totalLoss := float32(0)
	numTokens := batch * seqLen

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			offset := (b*seqLen + s) * vocabSize
			targetIdx := int(targetsData[b*seqLen+s])
			if targetIdx < 0 || targetIdx >= vocabSize {
				panic("target index out of range in crossEntropyLoss")
			}
			row := logitsData[offset : offset+vocabSize]

			// Numerical stability: subtract max before exp
			_, maxVal := argmax(row)

			sumExp := float32(0)
			for _, logit := range row {
				sumExp += ExpF32(logit - maxVal)
			}
			// log_prob = logit[target] - max - log(sum_exp)
			logProb := row[targetIdx] - maxVal - LogF32(sumExp)
			totalLoss -= logProb
		}
	}

	return totalLoss / float32(numTokens)
}

// crossEntropyGrad computes dL/d(logits) for cross-entropy loss.
//
//   grad[b, s, v] = (softmax(logits[b,s])[v] - one_hot(target[b,s])[v]) / N
//
// This is the standard softmax gradient: prob - target.
func crossEntropyGrad(logits, targets *Tensor) *Tensor {
	dims := logits.Shape().DimsRef()
	batch, seqLen, vocabSize := dims[0], dims[1], dims[2]
	numTokens := batch * seqLen

	grad := Zeros(logits.Shape(), F32)
	logitsData := logits.DataPtr()
	targetsData := targets.DataPtr()
	gradData := grad.DataPtr()

	rowBuffer := make([]float32, vocabSize)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			tokenIdx := b*seqLen + s
			offset := tokenIdx * vocabSize
			targetIdx := int(targetsData[tokenIdx])
			if targetIdx < 0 || targetIdx >= vocabSize {
				panic("target index out of range in crossEntropyGrad")
			}

			copy(rowBuffer, logitsData[offset:offset+vocabSize])
			softmaxInPlace(rowBuffer)
			copy(gradData[offset:offset+vocabSize], rowBuffer)
			// softmax(logits)[target] - 1.0 (the one-hot subtraction)
			gradData[offset+targetIdx] -= 1.0
		}
	}

	// Average over all tokens
	scale := 1.0 / float32(numTokens)
	for i := range gradData {
		gradData[i] *= scale
	}
	return grad
}

// clipTensorByGlobalNorm clips the tensor's L2 norm to clipNorm if it exceeds it.
// Modifies the tensor in-place. Returns the original (pre-clip) norm.
//
//   if ||t||_2 > clip_norm:  t = t * (clip_norm / ||t||_2)
func clipTensorByGlobalNorm(t *Tensor, clipNorm float32) float32 {
	if clipNorm <= 0 {
		return 0
	}
	data := t.DataPtr()
	sumSq := float32(0)
	for _, g := range data {
		sumSq += g * g
	}
	norm := SqrtF32(sumSq)
	if norm > clipNorm {
		scale := clipNorm / (norm + 1e-12) // epsilon prevents division by zero
		for i := range data {
			data[i] *= scale
		}
	}
	return norm
}

// TrainStep performs a single training step: forward, loss, backward, AdamW update.
//
// AdamW update rule per parameter:
//   m = beta1 * m + (1 - beta1) * g           -- first moment
//   v = beta2 * v + (1 - beta2) * g^2          -- second moment
//   m_hat = m / (1 - beta1^t)                  -- bias correction
//   v_hat = v / (1 - beta2^t)                  -- bias correction
//   w -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)
//
// The weight decay term is applied directly to w (decoupled, hence "AdamW"),
// not added to the gradient.
func (t *Trainer) TrainStep(input, targets *Tensor) float32 {
	t.step++

	// Zero all parameter gradients before forward/backward
	params := t.model.Parameters()
	for _, p := range params {
		p.ZeroGrad()
	}

	// Forward pass
	logits := t.model.Forward(input)
	loss := crossEntropyLoss(logits, targets)

	// Add auxiliary loss from MoE routers (load balancing)
	auxLoss := t.model.TotalAuxLoss(t.config.AuxAlpha)
	totalLoss := loss + auxLoss

	// Backward pass: computes and stores per-parameter gradients on param.Grad
	gradOutput := crossEntropyGrad(logits, targets)
	_ = t.model.Backward(gradOutput)

	// Global gradient norm clipping across all parameters
	globalNormSq := float32(0)
	for _, p := range params {
		if p.Grad != nil {
			for _, g := range p.Grad {
				globalNormSq += g * g
			}
		}
	}
	globalNorm := SqrtF32(globalNormSq)

	clipCoeff := float32(1.0)
	if t.config.GradClip > 0 && globalNorm > t.config.GradClip {
		clipCoeff = t.config.GradClip / (globalNorm + 1e-12)
	}

	// AdamW update using actual per-parameter gradients
	lr := t.GetLR()
	mCorr := 1.0 / (1 - PowF32(t.config.Beta1, float32(t.step)))
	vCorr := 1.0 / (1 - PowF32(t.config.Beta2, float32(t.step)))
	b1, b2, eps, wd := t.config.Beta1, t.config.Beta2, t.config.Eps, t.config.WeightDecay

	for i, param := range params {
		// Skip parameters that received no gradient in this backward pass.
		// Router gate weights get zero gradients (Router.Backward returns zeros)
		// so their Grad stays nil. Skipping them prevents AdamW momentum and
		// weight decay from drifting these weights, which would cause aux loss
		// to increase monotonically and dominate the total loss.
		if param.Grad == nil {
			continue
		}
		paramData := param.DataPtr()
		mData := t.states[i].M.DataPtr()
		vData := t.states[i].V.DataPtr()
		gradSlice := param.Grad

		for j := range paramData {
			grad := gradSlice[j] * clipCoeff
			mData[j] = b1*mData[j] + (1-b1)*grad
			vData[j] = b2*vData[j] + (1-b2)*grad*grad
			paramData[j] -= lr * (mData[j]*mCorr/(SqrtF32(vData[j]*vCorr)+eps) + wd*paramData[j])
		}
	}

	return totalLoss
}

// ---------------------------------------------------------------------------
// Activation Checkpointing
// ---------------------------------------------------------------------------

// CheckpointStorage stores intermediate activations for gradient checkpointing,
// which trades compute for memory by recomputing activations instead of storing all.
type CheckpointStorage struct {
	checkpoints map[int]*Tensor
	enabled     bool
}

// NewCheckpointStorage creates a checkpoint store.
func NewCheckpointStorage(enabled bool) *CheckpointStorage {
	return &CheckpointStorage{
		checkpoints: make(map[int]*Tensor),
		enabled:     enabled,
	}
}

// Enabled returns whether checkpointing is active.
func (s *CheckpointStorage) Enabled() bool { return s.enabled }

// Save stores an activation tensor at the given block index.
func (s *CheckpointStorage) Save(blockIdx int, t *Tensor) {
	if s.enabled {
		s.checkpoints[blockIdx] = t
	}
}

// Get retrieves the stored activation for a block index.
func (s *CheckpointStorage) Get(blockIdx int) *Tensor {
	return s.checkpoints[blockIdx]
}

// Clear removes all stored checkpoints.
func (s *CheckpointStorage) Clear() { s.checkpoints = make(map[int]*Tensor) }

// Len returns the number of stored checkpoints.
func (s *CheckpointStorage) Len() int { return len(s.checkpoints) }

// CheckpointContext manages which blocks to checkpoint based on a segment size.
// Every SegmentSize-th block saves its activation; during backward, intermediate
// blocks are recomputed from the nearest checkpoint.
type CheckpointContext struct {
	Storage     *CheckpointStorage
	SegmentSize int
}

// NewCheckpointContext creates a context that checkpoints every segmentSize blocks.
func NewCheckpointContext(segmentSize int) *CheckpointContext {
	size := segmentSize
	if size < 1 {
		size = 1
	}
	return &CheckpointContext{
		Storage:     NewCheckpointStorage(segmentSize > 0),
		SegmentSize: size,
	}
}

// DisabledCheckpointContext creates a context with checkpointing disabled.
func DisabledCheckpointContext() *CheckpointContext {
	return &CheckpointContext{
		Storage:     NewCheckpointStorage(false),
		SegmentSize: 1,
	}
}

// ShouldCheckpoint returns true if blockIdx is a checkpoint boundary.
func (c *CheckpointContext) ShouldCheckpoint(blockIdx int) bool {
	return c.Storage.Enabled() && (blockIdx%c.SegmentSize == 0)
}

// MaybeSave conditionally stores an activation if this block is a checkpoint boundary.
func (c *CheckpointContext) MaybeSave(blockIdx int, t *Tensor) {
	if c.ShouldCheckpoint(blockIdx) {
		c.Storage.Save(blockIdx, t)
	}
}

// GetCheckpoint retrieves the nearest checkpoint at or before blockIdx.
func (c *CheckpointContext) GetCheckpoint(blockIdx int) *Tensor {
	cpIdx := (blockIdx / c.SegmentSize) * c.SegmentSize
	return c.Storage.Get(cpIdx)
}

// Clear removes all stored checkpoints.
func (c *CheckpointContext) Clear() { c.Storage.Clear() }

// ---------------------------------------------------------------------------
// Mixed Precision / Loss Scaling
// ---------------------------------------------------------------------------

// LossScaleMode determines whether loss scaling is static or dynamic.
type LossScaleMode int

const (
	// LossScaleStatic uses a fixed loss scale value.
	LossScaleStatic LossScaleMode = iota
	// LossScaleDynamic adjusts loss scale based on overflow detection.
	LossScaleDynamic
)

// LossScaler manages loss scaling for mixed-precision training.
// In FP16 training, gradients are scaled up to prevent underflow, then
// unscaled before the optimizer step. Dynamic scaling adjusts the scale
// factor based on whether gradient overflow is detected.
type LossScaler struct {
	mode          LossScaleMode
	scale         float32
	scaleFactor   float32 // multiplicative factor for scale adjustments
	scaleWindow   int     // steps without overflow before scale increase
	growthTracker int
	overflow      bool
}

// NewLossScaler creates a loss scaler with the given mode and parameters.
func NewLossScaler(mode LossScaleMode, initScale, scaleFactor float32, scaleWindow int) *LossScaler {
	return &LossScaler{
		mode:        mode,
		scale:       initScale,
		scaleFactor: scaleFactor,
		scaleWindow: scaleWindow,
	}
}

// StaticLossScaler creates a static scaler with a fixed scale value.
func StaticLossScaler(scale float32) *LossScaler {
	return NewLossScaler(LossScaleStatic, scale, 2.0, 2000)
}

// DynamicLossScaler creates a dynamic scaler starting at 65536.
func DynamicLossScaler() *LossScaler {
	return NewLossScaler(LossScaleDynamic, 65536.0, 2.0, 2000)
}

// Scale returns the current loss scale value.
func (s *LossScaler) Scale() float32 { return s.scale }

// ScaleLoss multiplies the loss by the current scale (for scaled backward pass).
func (s *LossScaler) ScaleLoss(loss float32) float32 { return loss * s.scale }

// UnscaleGrads divides a gradient by the scale (restoring true gradient magnitude).
func (s *LossScaler) UnscaleGrads(grad float32) float32 { return grad / s.scale }

// CheckOverflow detects NaN/Inf in gradients, indicating FP16 overflow.
// Uses the NaN != NaN property for NaN detection.
func (s *LossScaler) CheckOverflow(grads []float32) bool {
	s.overflow = false
	for _, g := range grads {
		if g != g || g > 3.4e38 || g < -3.4e38 {
			s.overflow = true
			break
		}
	}
	return s.overflow
}

// Update adjusts the loss scale after each step.
// On overflow: scale /= factor, reset growth counter.
// On no overflow for scaleWindow consecutive steps: scale *= factor.
func (s *LossScaler) Update() {
	if s.mode == LossScaleStatic {
		return
	}
	if s.overflow {
		s.scale /= s.scaleFactor
		s.growthTracker = 0
		s.overflow = false
		return
	}
	s.growthTracker++
	if s.growthTracker >= s.scaleWindow {
		s.scale *= s.scaleFactor
		s.growthTracker = 0
	}
}

// ShouldSkipStep returns true if the current step had gradient overflow
// and the optimizer update should be skipped.
func (s *LossScaler) ShouldSkipStep() bool { return s.overflow }

// MixedPrecisionConfig defines which layers use FP32 vs FP16 compute.
type MixedPrecisionConfig struct {
	Enabled      bool
	ComputeDType DType
	LossScale    LossScaleMode
	FP32Layers   []string // layer names that must remain in FP32 for stability
}

// DefaultMixedPrecisionConfig returns a config with mixed precision disabled.
func DefaultMixedPrecisionConfig() MixedPrecisionConfig {
	return MixedPrecisionConfig{
		Enabled:      false,
		ComputeDType: F16,
		LossScale:    LossScaleDynamic,
		FP32Layers:   []string{"final_norm", "lm_head"},
	}
}

// FP16MixedPrecisionConfig returns a config with FP16 mixed precision enabled.
func FP16MixedPrecisionConfig() MixedPrecisionConfig {
	cfg := DefaultMixedPrecisionConfig()
	cfg.Enabled = true
	return cfg
}

// IsFP32Layer checks if a layer name matches any FP32 layer pattern.
func (c MixedPrecisionConfig) IsFP32Layer(name string) bool {
	for _, s := range c.FP32Layers {
		if strings.Contains(name, s) {
			return true
		}
	}
	return false
}

// MasterWeights stores FP32 copies of weights for mixed-precision training.
// During training, compute happens in FP16 but the optimizer updates are
// applied to these FP32 master copies to prevent precision loss.
type MasterWeights struct {
	weights []*Tensor
}

// NewMasterWeights creates FP32 master weight copies for all parameters.
func NewMasterWeights(params []*Tensor) *MasterWeights {
	w := make([]*Tensor, len(params))
	for i, p := range params {
		w[i] = Zeros(p.Shape(), F32)
	}
	return &MasterWeights{weights: w}
}

// Weights returns the master weight tensors.
func (m *MasterWeights) Weights() []*Tensor { return m.weights }

// Len returns the number of master weight tensors.
func (m *MasterWeights) Len() int { return len(m.weights) }
