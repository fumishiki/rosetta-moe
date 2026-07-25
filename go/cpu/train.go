// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

package cpu

import "strings"

// TrainConfig holds optimizer and training hyperparameters.
type TrainConfig struct {
	LR           float32
	Beta1        float32
	Beta2        float32
	Eps          float32
	WeightDecay  float32
	GradClip     float32
	WarmupSteps  int
	TotalSteps   int
	AuxAlpha     float32
	ZLossWeight  float32
	RoutingMode  RoutingMode
	BiasGamma    float32
	ReLULambdaL1 float32
	ReLUTargetK  int
}

// DefaultTrainConfig returns standard training hyperparameters.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		LR:           1e-4,
		Beta1:        0.9,
		Beta2:        0.95,
		Eps:          1e-8,
		WeightDecay:  0.1,
		GradClip:     0.5,
		WarmupSteps:  1000,
		TotalSteps:   100000,
		AuxAlpha:     0.01,
		ZLossWeight:  0.05,
		RoutingMode:  TopKMode,
		BiasGamma:    0.001,
		ReLULambdaL1: 0.01,
		ReLUTargetK:  2,
	}
}

// AdamWState holds the first and second moment estimates for one parameter tensor.
type AdamWState struct {
	M *Tensor
	V *Tensor
}

// Trainer encapsulates the model, optimizer state, and LR schedule.
type Trainer struct {
	Model  *MoETransformer
	Config TrainConfig
	step   int
	Params []*Tensor
	states []AdamWState
	ceRowBuf     []float32
	gradLogits   *Tensor
	beta1PowStep float32
	beta2PowStep float32
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
	return &Trainer{
		Model:        m,
		Config:       cfg,
		Params:       params,
		states:       states,
		beta1PowStep: 1.0,
		beta2PowStep: 1.0,
	}
}

// GetLR computes the current learning rate using linear warmup + cosine decay.
func (t *Trainer) GetLR() float32 {
	if t.step < t.Config.WarmupSteps {
		return t.Config.LR * float32(t.step) / float32(t.Config.WarmupSteps)
	}
	progress := float32(t.step-t.Config.WarmupSteps) / float32(t.Config.TotalSteps-t.Config.WarmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	minLR := t.Config.LR * 0.1
	return minLR + 0.5*(t.Config.LR-minLR)*(1.0+CosF32(3.1415927*progress))
}

// Step returns the current training step count.
func (t *Trainer) Step() int { return t.step }

// CrossEntropyLossGrad computes both mean cross-entropy loss and dL/d(logits) in a single pass.
func (t *Trainer) CrossEntropyLossGrad(logits, targets *Tensor) (float32, *Tensor) {
	dims := logits.Shape().DimsRef()
	batch, seqLen, vocabSize := dims[0], dims[1], dims[2]
	numTokens := batch * seqLen

	if t.gradLogits == nil || !t.gradLogits.Shape().Equal(logits.Shape()) {
		t.gradLogits = New(logits.Shape(), F32)
	}
	gradData := t.gradLogits.DataPtr()
	logitsData := logits.DataPtr()
	targetsData := targets.DataPtr()

	if cap(t.ceRowBuf) < vocabSize {
		t.ceRowBuf = make([]float32, vocabSize)
	}
	rowBuffer := t.ceRowBuf[:vocabSize]

	totalLoss := float32(0)
	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		offset := tokenIdx * vocabSize
		targetIdx := int(targetsData[tokenIdx])
		if targetIdx < 0 || targetIdx >= vocabSize {
			panic("target index out of range in CrossEntropyLossGrad")
		}

		copy(rowBuffer, logitsData[offset:offset+vocabSize])
		softmaxInPlace(rowBuffer)
		copy(gradData[offset:offset+vocabSize], rowBuffer)

		p := rowBuffer[targetIdx]
		if p < 1e-12 {
			p = 1e-12
		}
		totalLoss -= LogF32(p)

		gradData[offset+targetIdx] -= 1.0
	}

	scale := 1.0 / float32(numTokens)
	for i := range gradData {
		gradData[i] *= scale
	}
	return totalLoss * scale, t.gradLogits
}

// CrossEntropyLoss computes the mean cross-entropy loss.
func CrossEntropyLoss(logits, targets *Tensor) float32 {
	tmp := Trainer{}
	loss, _ := tmp.CrossEntropyLossGrad(logits, targets)
	return loss
}

// CrossEntropyGrad computes the gradient of cross-entropy loss.
func CrossEntropyGrad(logits, targets *Tensor) *Tensor {
	tmp := Trainer{}
	_, grad := tmp.CrossEntropyLossGrad(logits, targets)
	return grad
}

// ClipTensorByGlobalNorm clips the tensor's L2 norm.
func ClipTensorByGlobalNorm(t *Tensor, clipNorm float32) float32 {
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
		scale := clipNorm / (norm + 1e-12)
		for i := range data {
			data[i] *= scale
		}
	}
	return norm
}

// TrainStepFromLogits performs a training step using pre-computed logits.
func (t *Trainer) TrainStepFromLogits(logits, targets *Tensor) float32 {
	t.step++

	params := t.Params
	for _, p := range params {
		p.ZeroGrad()
	}

	loss, gradOutput := t.CrossEntropyLossGrad(logits, targets)

	_ = t.Model.Backward(gradOutput)

	var auxLoss, zLoss, reluL1 float32
	switch t.Config.RoutingMode {
	case TopKMode:
		auxLoss = t.Model.ApplyAuxLoss(t.Config.AuxAlpha)
		zLoss = t.Model.ApplyZLoss(t.Config.ZLossWeight)
	case BiasFreeMode:
		zLoss = t.Model.ApplyZLoss(t.Config.ZLossWeight)
	case ReLUMode:
		reluL1 = t.Model.ApplyReLUL1Loss()
	}

	totalLoss := loss + auxLoss + zLoss + reluL1

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
	if t.Config.GradClip > 0 && globalNorm > t.Config.GradClip {
		clipCoeff = t.Config.GradClip / (globalNorm + 1e-12)
	}

	lr := t.GetLR()
	t.beta1PowStep *= t.Config.Beta1
	t.beta2PowStep *= t.Config.Beta2
	mCorr := 1.0 / (1 - t.beta1PowStep)
	vCorr := 1.0 / (1 - t.beta2PowStep)
	b1, b2, eps, wd := t.Config.Beta1, t.Config.Beta2, t.Config.Eps, t.Config.WeightDecay

	for i, param := range params {
		if param.Grad == nil {
			continue
		}
		paramData := param.DataPtr()
		mData := t.states[i].M.DataPtr()
		vData := t.states[i].V.DataPtr()
		gradSlice := param.Grad

		if clipCoeff != 1.0 {
			for j := range paramData {
				grad := gradSlice[j] * clipCoeff
				mData[j] = b1*mData[j] + (1-b1)*grad
				vData[j] = b2*vData[j] + (1-b2)*grad*grad
				paramData[j] -= lr * (mData[j]*mCorr/(SqrtF32(vData[j]*vCorr)+eps) + wd*paramData[j])
			}
		} else {
			for j := range paramData {
				grad := gradSlice[j]
				mData[j] = b1*mData[j] + (1-b1)*grad
				vData[j] = b2*vData[j] + (1-b2)*grad*grad
				paramData[j] -= lr * (mData[j]*mCorr/(SqrtF32(vData[j]*vCorr)+eps) + wd*paramData[j])
			}
		}
	}

	switch t.Config.RoutingMode {
	case BiasFreeMode:
		t.Model.UpdateRoutingBiases(t.Config.BiasGamma)
	case ReLUMode:
		avgActive := t.Model.AvgActiveExperts()
		targetK := float32(t.Config.ReLUTargetK)
		if avgActive < targetK*0.9 {
			t.Config.ReLULambdaL1 *= 0.99
		} else if avgActive > targetK*1.1 {
			t.Config.ReLULambdaL1 *= 1.01
		}
		t.Model.SetRoutingMode(ReLUMode, t.Config.ReLULambdaL1)
	}

	return totalLoss
}

// TrainStep performs a single training step: forward, loss, backward, AdamW update.
func (t *Trainer) TrainStep(input, targets *Tensor) float32 {
	logits := t.Model.Forward(input)
	return t.TrainStepFromLogits(logits, targets)
}

// ---------------------------------------------------------------------------
// Activation Checkpointing
// ---------------------------------------------------------------------------

// CheckpointStorage stores intermediate activations for gradient checkpointing.
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

// CheckpointContext manages which blocks to checkpoint.
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

// MaybeSave conditionally stores an activation.
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
	LossScaleStatic LossScaleMode = iota
	LossScaleDynamic
)

// LossScaler manages loss scaling for mixed-precision training.
type LossScaler struct {
	mode          LossScaleMode
	scale         float32
	scaleFactor   float32
	scaleWindow   int
	growthTracker int
	overflow      bool
}

// NewLossScaler creates a loss scaler.
func NewLossScaler(mode LossScaleMode, initScale, scaleFactor float32, scaleWindow int) *LossScaler {
	return &LossScaler{mode: mode, scale: initScale, scaleFactor: scaleFactor, scaleWindow: scaleWindow}
}

// StaticLossScaler creates a static scaler.
func StaticLossScaler(scale float32) *LossScaler {
	return NewLossScaler(LossScaleStatic, scale, 2.0, 2000)
}

// DynamicLossScaler creates a dynamic scaler starting at 65536.
func DynamicLossScaler() *LossScaler {
	return NewLossScaler(LossScaleDynamic, 65536.0, 2.0, 2000)
}

// Scale returns the current loss scale value.
func (s *LossScaler) Scale() float32 { return s.scale }

// ScaleLoss multiplies the loss by the current scale.
func (s *LossScaler) ScaleLoss(loss float32) float32 { return loss * s.scale }

// UnscaleGrads divides a gradient by the scale.
func (s *LossScaler) UnscaleGrads(grad float32) float32 { return grad / s.scale }

// CheckOverflow detects NaN/Inf in gradients.
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

// ShouldSkipStep returns true if the current step had gradient overflow.
func (s *LossScaler) ShouldSkipStep() bool { return s.overflow }

// MixedPrecisionConfig defines which layers use FP32 vs FP16 compute.
type MixedPrecisionConfig struct {
	Enabled      bool
	ComputeDType DType
	LossScale    LossScaleMode
	FP32Layers   []string
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
type MasterWeights struct {
	weights []*Tensor
}

// NewMasterWeights creates FP32 master weight copies.
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
