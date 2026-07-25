// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build !darwin

package gpu

// MetalContext is a stub for non-macOS platforms.
type MetalContext struct{}

// MetalTensor is a stub for non-macOS platforms.
type MetalTensor struct{}

// NewMetalContext returns nil on non-macOS platforms.
func NewMetalContext() *MetalContext {
	return nil
}

// LoadShaderLibrary is a no-op on non-macOS platforms.
func (ctx *MetalContext) LoadShaderLibrary(path string) error {
	return nil
}

// LoadShaderSource is a no-op on non-macOS platforms.
func (ctx *MetalContext) LoadShaderSource(source string) error {
	return nil
}

// LoadRequiredShaders is a no-op on non-macOS platforms.
func (ctx *MetalContext) LoadRequiredShaders(shadersDir string) error {
	return nil
}

// NewTensorU32 returns nil on non-macOS platforms.
func (ctx *MetalContext) NewTensorU32(val uint32) *MetalTensor {
	return nil
}

// NewTensorF32Scalar returns nil on non-macOS platforms.
func (ctx *MetalContext) NewTensorF32Scalar(val float32) *MetalTensor {
	return nil
}

// NewTensor returns nil on non-macOS platforms.
func (ctx *MetalContext) NewTensor(data []float32, shape ...int) *MetalTensor {
	return nil
}

// NewTensorZeros returns nil on non-macOS platforms.
func (ctx *MetalContext) NewTensorZeros(shape ...int) *MetalTensor {
	return nil
}

// Data returns nil on non-macOS platforms.
func (t *MetalTensor) Data() []float32 {
	return nil
}

// DataCopy returns nil on non-macOS platforms.
func (t *MetalTensor) DataCopy() []float32 {
	return nil
}

// Matmul is a no-op on non-macOS platforms.
func (ctx *MetalContext) Matmul(a, b, c *MetalTensor, M, N, K int) {}

// DispatchKernel is a no-op on non-macOS platforms.
func (ctx *MetalContext) DispatchKernel(fnName string, buffers []*MetalTensor, gridSize, threadgroupSize [3]int) error {
	return nil
}

// Close is a no-op on non-macOS platforms.
func (ctx *MetalContext) Close() {}

// Release is a no-op on non-macOS platforms.
func (t *MetalTensor) Release() {}

// MetalAvailable returns false on non-macOS platforms.
func MetalAvailable() bool {
	return false
}

// UploadModel returns nil on non-macOS platforms.
func (ctx *MetalContext) UploadModel(spec *ModelSpec) *GpuModel {
	return nil
}

// ReleaseModel is a no-op on non-macOS platforms.
func (m *GpuModel) ReleaseModel() {}

// GpuForward returns nil on non-macOS platforms.
func GpuForward(ctx *MetalContext, model *GpuModel, inputIDs []float32, batch, seqLen int) *MetalTensor {
	return nil
}

// GpuForwardTensor returns nil on non-macOS platforms.
func GpuForwardTensor(ctx *MetalContext, model *GpuModel, inputIDs *MetalTensor, batch, seqLen int) *MetalTensor {
	return nil
}

// Stub types needed for non-darwin compilation.

// GpuLinearWeights stub.
type GpuLinearWeights struct {
	Weight      *MetalTensor
	WeightT     *MetalTensor
	InFeatures  int
	OutFeatures int
}

// GpuSwiGLUWeights stub.
type GpuSwiGLUWeights struct {
	WGate, WUp, WDown GpuLinearWeights
}

// GpuRMSNormWeights stub.
type GpuRMSNormWeights struct {
	Weight *MetalTensor
	Eps    float32
}

// GpuMoEWeights stub.
type GpuMoEWeights struct {
	RouterGate GpuLinearWeights
	Experts    []GpuSwiGLUWeights
	NExperts   int
	TopK       int
}

// GpuBlockWeights stub.
type GpuBlockWeights struct {
	AttnNorm                       GpuRMSNormWeights
	FfnNorm                        GpuRMSNormWeights
	AttnWQ, AttnWK, AttnWV, AttnWO GpuLinearWeights
	Moe                            GpuMoEWeights
}

// GpuModel stub.
type GpuModel struct {
	Embedding *MetalTensor
	Blocks    []GpuBlockWeights
	FinalNorm GpuRMSNormWeights
	LmHead    GpuLinearWeights
	VocabSize int
	HiddenDim int
	FFNDim    int
	NHeads    int
	NKVHeads  int
	HeadDim   int
}

// ModelSpec stub.
type ModelSpec struct {
	VocabSize       int
	HiddenDim       int
	FFNDim          int
	NLayers         int
	NHeads          int
	NKVHeads        int
	HeadDim         int
	NExperts        int
	TopK            int
	EmbeddingWeight []float32
	FinalNormWeight []float32
	FinalNormEps    float32
	LmHeadWeight    []float32
	Blocks          []BlockSpec
}

// BlockSpec stub.
type BlockSpec struct {
	AttnNormWeight   []float32
	AttnNormEps      float32
	FfnNormWeight    []float32
	FfnNormEps       float32
	WQWeight         []float32
	WQIn, WQOut      int
	WKWeight         []float32
	WKIn, WKOut      int
	WVWeight         []float32
	WVIn, WVOut      int
	WOWeight         []float32
	WOIn, WOOut      int
	RouterGateWeight []float32
	RouterGateIn     int
	RouterGateOut    int
	Experts          []ExpertSpec
}

// ExpertSpec stub.
type ExpertSpec struct {
	WGateWeight       []float32
	WGateIn, WGateOut int
	WUpWeight         []float32
	WUpIn, WUpOut     int
	WDownWeight       []float32
	WDownIn, WDownOut int
}
