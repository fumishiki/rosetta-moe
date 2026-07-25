// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build !darwin

package nn

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

// GpuForward returns nil on non-macOS platforms.
func GpuForward(ctx *MetalContext, model *MoETransformer, input *Tensor) *Tensor {
	return nil
}

// GpuTrainStep returns 0 on non-macOS platforms.
func GpuTrainStep(ctx *MetalContext, trainer *Trainer, input, targets *Tensor) float32 {
	return 0
}
