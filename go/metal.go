// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build darwin

package nn

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework CoreGraphics
#cgo CFLAGS: -x objective-c -fobjc-arc
#include <stdlib.h>
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

// MetalContext holds a Metal device, command queue, and optional shader library.
type MetalContext struct {
	device    unsafe.Pointer
	queue     unsafe.Pointer
	shaderLib unsafe.Pointer
}

// MetalTensor wraps a Metal buffer with shape information.
type MetalTensor struct {
	buffer unsafe.Pointer
	shape  []int
	len    int
}

// NewMetalContext creates a Metal device and command queue.
// Returns nil if Metal is not available (e.g., non-Apple platform or no GPU).
func NewMetalContext() *MetalContext {
	dev := C.metal_create_device()
	if dev == nil {
		return nil
	}
	queue := C.metal_create_command_queue(dev)
	if queue == nil {
		C.metal_release(dev)
		return nil
	}
	return &MetalContext{
		device: dev,
		queue:  queue,
	}
}

// LoadShaderLibrary loads a compiled .metallib file from the given path.
// Returns error if the file cannot be loaded.
func (ctx *MetalContext) LoadShaderLibrary(path string) error {
	if ctx == nil || ctx.device == nil {
		return fmt.Errorf("invalid MetalContext")
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	lib := C.metal_load_library(ctx.device, cPath)
	if lib == nil {
		return fmt.Errorf("failed to load Metal library: %s", path)
	}
	if ctx.shaderLib != nil {
		C.metal_release(ctx.shaderLib)
	}
	ctx.shaderLib = lib
	return nil
}

// LoadShaderSource compiles and loads Metal shader source code at runtime.
// Returns error if compilation fails.
func (ctx *MetalContext) LoadShaderSource(source string) error {
	if ctx == nil || ctx.device == nil {
		return fmt.Errorf("invalid MetalContext")
	}
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	lib := C.metal_load_library_from_source(ctx.device, cSource)
	if lib == nil {
		return fmt.Errorf("failed to compile Metal shader source")
	}
	if ctx.shaderLib != nil {
		C.metal_release(ctx.shaderLib)
	}
	ctx.shaderLib = lib
	return nil
}

// LoadRequiredShaders loads and compiles all Metal shaders needed for GPU forward pass.
// shadersDir: path to directory containing .metal files (e.g., "../shaders").
func (ctx *MetalContext) LoadRequiredShaders(shadersDir string) error {
	files := []string{"rmsnorm.metal", "silu.metal", "elementwise.metal"}
	var combined strings.Builder
	for _, f := range files {
		content, err := os.ReadFile(filepath.Join(shadersDir, f))
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", f, err)
		}
		combined.Write(content)
		combined.WriteByte('\n')
	}
	return ctx.LoadShaderSource(combined.String())
}

// NewTensor creates a Metal buffer from Go float32 slice.
// Data is copied to GPU-accessible unified memory (MTLResourceStorageModeShared).
func (ctx *MetalContext) NewTensor(data []float32, shape ...int) *MetalTensor {
	if ctx == nil || ctx.device == nil || len(data) == 0 {
		return nil
	}
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	buf := C.metal_create_buffer(
		ctx.device,
		unsafe.Pointer(&data[0]),
		C.size_t(len(data)*4),
	)
	if buf == nil {
		return nil
	}
	return &MetalTensor{
		buffer: buf,
		shape:  shape,
		len:    len(data),
	}
}

// NewTensorZeros creates a zero-initialized Metal buffer.
func (ctx *MetalContext) NewTensorZeros(shape ...int) *MetalTensor {
	numel := 1
	for _, s := range shape {
		numel *= s
	}
	data := make([]float32, numel)
	return ctx.NewTensor(data, shape...)
}

// Data copies buffer contents back to Go slice (GPU → CPU).
func (t *MetalTensor) Data() []float32 {
	if t == nil || t.buffer == nil || t.len == 0 {
		return nil
	}
	ptr := C.metal_buffer_contents(t.buffer)
	if ptr == nil {
		return nil
	}
	// Reconstruct slice from raw pointer
	data := make([]float32, t.len)
	src := (*[1 << 30]float32)(ptr)[:t.len:t.len]
	copy(data, src)
	return data
}

// Matmul performs matrix multiplication using MPS: C = A @ B.
// A: [M, K], B: [K, N], C: [M, N] (row-major).
func (ctx *MetalContext) Matmul(a, b, c *MetalTensor, M, N, K int) {
	if ctx == nil || ctx.device == nil || ctx.queue == nil {
		return
	}
	if a == nil || b == nil || c == nil {
		return
	}
	C.metal_mps_matmul(
		ctx.device,
		ctx.queue,
		a.buffer,
		b.buffer,
		c.buffer,
		C.int(M),
		C.int(N),
		C.int(K),
	)
}

// DispatchKernel runs a custom Metal compute kernel from the loaded shader library.
// fnName: kernel function name (e.g., "matmul_kernel")
// buffers: input/output buffers
// gridSize: [x, y, z] dimensions
// threadgroupSize: [x, y, z] dimensions
func (ctx *MetalContext) DispatchKernel(fnName string, buffers []*MetalTensor, gridSize, threadgroupSize [3]int) error {
	if ctx == nil || ctx.shaderLib == nil || ctx.queue == nil {
		return fmt.Errorf("invalid MetalContext or missing shader library")
	}

	// Create function
	cFnName := C.CString(fnName)
	defer C.free(unsafe.Pointer(cFnName))
	fn := C.metal_create_function(ctx.shaderLib, cFnName)
	if fn == nil {
		return fmt.Errorf("failed to create Metal function: %s", fnName)
	}
	defer C.metal_release(fn)

	// Create pipeline
	pipeline := C.metal_create_pipeline(ctx.device, fn)
	if pipeline == nil {
		return fmt.Errorf("failed to create Metal pipeline for: %s", fnName)
	}
	defer C.metal_release(pipeline)

	// Prepare buffer pointers
	cBufs := make([]unsafe.Pointer, len(buffers))
	for i, t := range buffers {
		if t != nil {
			cBufs[i] = t.buffer
		}
	}

	// Dispatch
	var bufPtr *unsafe.Pointer
	if len(cBufs) > 0 {
		bufPtr = &cBufs[0]
	}
	C.metal_dispatch_kernel(
		ctx.queue,
		pipeline,
		bufPtr,
		C.int(len(cBufs)),
		C.int(gridSize[0]),
		C.int(gridSize[1]),
		C.int(gridSize[2]),
		C.int(threadgroupSize[0]),
		C.int(threadgroupSize[1]),
		C.int(threadgroupSize[2]),
	)

	return nil
}

// Close releases Metal resources.
func (ctx *MetalContext) Close() {
	if ctx == nil {
		return
	}
	if ctx.shaderLib != nil {
		C.metal_release(ctx.shaderLib)
		ctx.shaderLib = nil
	}
	if ctx.queue != nil {
		C.metal_release(ctx.queue)
		ctx.queue = nil
	}
	if ctx.device != nil {
		C.metal_release(ctx.device)
		ctx.device = nil
	}
}

// Release frees the Metal buffer.
func (t *MetalTensor) Release() {
	if t != nil && t.buffer != nil {
		C.metal_release(t.buffer)
		t.buffer = nil
	}
}

// MetalAvailable checks if Metal is available on the current platform.
func MetalAvailable() bool {
	ctx := NewMetalContext()
	if ctx == nil {
		return false
	}
	ctx.Close()
	return true
}

// NewTensorU32 creates a Metal buffer containing a single uint32 value.
// Used for passing scalar parameters to Metal kernels.
func (ctx *MetalContext) NewTensorU32(val uint32) *MetalTensor {
	buf := C.metal_create_buffer(ctx.device, unsafe.Pointer(&val), C.size_t(4))
	if buf == nil {
		return nil
	}
	return &MetalTensor{buffer: buf, shape: []int{1}, len: 1}
}

// NewTensorF32Scalar creates a Metal buffer containing a single float32 value.
// Used for passing scalar parameters to Metal kernels.
func (ctx *MetalContext) NewTensorF32Scalar(val float32) *MetalTensor {
	buf := C.metal_create_buffer(ctx.device, unsafe.Pointer(&val), C.size_t(4))
	if buf == nil {
		return nil
	}
	return &MetalTensor{buffer: buf, shape: []int{1}, len: 1}
}

// gpuLinear computes Y = X @ W^T using MPS matmul.
// Weight: [outFeatures, inFeatures], Input: [batchSeq, inFeatures], Output: [batchSeq, outFeatures].
func gpuLinear(ctx *MetalContext, input *MetalTensor, weight *Tensor) *MetalTensor {
	wShape := weight.Shape()
	wDims := wShape.DimsRef()
	outFeatures := wDims[0]
	inFeatures := wDims[len(wDims)-1]
	batchSeq := input.len / inFeatures

	// Transpose weight [out, in] → [in, out] for MPS
	wData := weight.Data()
	wT := make([]float32, inFeatures*outFeatures)
	for r := 0; r < outFeatures; r++ {
		for c := 0; c < inFeatures; c++ {
			wT[c*outFeatures+r] = wData[r*inFeatures+c]
		}
	}
	wGpu := ctx.NewTensor(wT, inFeatures, outFeatures)
	out := ctx.NewTensorZeros(batchSeq, outFeatures)

	ctx.Matmul(input, wGpu, out, batchSeq, outFeatures, inFeatures)
	wGpu.Release()
	return out
}

// gpuRMSNorm dispatches the rmsnorm Metal kernel.
// Kernel uses threadgroup reduction, one threadgroup per row.
func gpuRMSNorm(ctx *MetalContext, input *MetalTensor, weightData []float32, nRows, hiddenDim int, eps float32) *MetalTensor {
	weightGpu := ctx.NewTensor(weightData, hiddenDim)
	out := ctx.NewTensorZeros(nRows, hiddenDim)
	dimBuf := ctx.NewTensorU32(uint32(hiddenDim))
	epsBuf := ctx.NewTensorF32Scalar(eps)

	// rmsnorm kernel: nRows threadgroups × 256 threads per threadgroup
	ctx.DispatchKernel("rmsnorm",
		[]*MetalTensor{input, out, weightGpu, dimBuf, epsBuf},
		[3]int{nRows * 256, 1, 1},
		[3]int{256, 1, 1},
	)

	weightGpu.Release()
	dimBuf.Release()
	epsBuf.Release()
	return out
}

// gpuSilu dispatches the silu Metal kernel (elementwise SiLU activation).
func gpuSilu(ctx *MetalContext, input *MetalTensor) *MetalTensor {
	n := input.len
	out := ctx.NewTensorZeros(input.shape...)
	ctx.DispatchKernel("silu",
		[]*MetalTensor{input, out},
		[3]int{n, 1, 1},
		[3]int{256, 1, 1},
	)
	return out
}

// gpuAdd dispatches the elementwise add Metal kernel.
func gpuAdd(ctx *MetalContext, a, b *MetalTensor) *MetalTensor {
	out := ctx.NewTensorZeros(a.shape...)
	ctx.DispatchKernel("add",
		[]*MetalTensor{a, b, out},
		[3]int{a.len, 1, 1},
		[3]int{256, 1, 1},
	)
	return out
}

// gpuMul dispatches the elementwise mul Metal kernel.
func gpuMul(ctx *MetalContext, a, b *MetalTensor) *MetalTensor {
	out := ctx.NewTensorZeros(a.shape...)
	ctx.DispatchKernel("mul",
		[]*MetalTensor{a, b, out},
		[3]int{a.len, 1, 1},
		[3]int{256, 1, 1},
	)
	return out
}

// GpuForward performs layer-by-layer GPU forward pass for the MoE Transformer.
// Hybrid approach: GPU for embeddings/norms/linear/SwiGLU, CPU for attention/routing.
// M1 unified memory enables zero-copy CPU↔GPU data transfer.
func GpuForward(ctx *MetalContext, model *MoETransformer, input *Tensor) *Tensor {
	cfg := model.Config()
	dims := input.Shape().DimsRef()
	batch := dims[0]
	seqLen := dims[1]
	hidden := cfg.HiddenDim
	batchSeq := batch * seqLen

	// 1. Embedding (CPU)
	embOut := model.embedding.Forward(input)
	embData := embOut.Data()
	// Reshape to [batchSeq, hidden] for GPU ops
	x := ctx.NewTensor(embData, batchSeq, hidden)

	// 2. Transformer blocks
	for _, block := range model.blocks {
		eps := block.attnNorm.eps

		// RMSNorm (attn)
		normed := gpuRMSNorm(ctx, x, block.attnNorm.weight.Data(), batchSeq, hidden, eps)

		// Attention (CPU fallback — RoPE requires complex indexing)
		normedData := normed.Data()
		normedCPU := FromSlice(normedData, NewShape(batch, seqLen, hidden))
		attnOut := block.attention.Forward(normedCPU)
		attnGpu := ctx.NewTensor(attnOut.Data(), batchSeq, hidden)

		// Residual
		xNew := gpuAdd(ctx, x, attnGpu)
		x.Release()
		attnGpu.Release()
		normed.Release()
		x = xNew

		// RMSNorm (ffn)
		normed2 := gpuRMSNorm(ctx, x, block.ffnNorm.weight.Data(), batchSeq, hidden, eps)

		// MoE (hybrid: routing CPU, expert SwiGLU GPU)
		normed2Data := normed2.Data()
		normed2CPU := FromSlice(normed2Data, NewShape(batch, seqLen, hidden))
		topK := block.moe.topK

		// Routing on CPU
		weights, indices := block.moe.router.Forward(normed2CPU)
		wData := weights.Data()

		// Build inverted index: expert -> list of tokens
		nExperts := len(block.moe.experts)
		expertTokens := make([][]int, nExperts)
		expertWeightIdx := make([][]int, nExperts)
		for i := range expertTokens {
			expertTokens[i] = nil
			expertWeightIdx[i] = nil
		}
		for t := 0; t < batchSeq; t++ {
			for k := 0; k < topK; k++ {
				eIdx := indices[t][k]
				expertTokens[eIdx] = append(expertTokens[eIdx], t)
				expertWeightIdx[eIdx] = append(expertWeightIdx[eIdx], k)
			}
		}

		// Process each expert with GPU SwiGLU
		moeOutData := make([]float32, batchSeq*hidden)
		for eIdx := 0; eIdx < nExperts; eIdx++ {
			tokens := expertTokens[eIdx]
			if len(tokens) == 0 {
				continue
			}
			nTok := len(tokens)

			// Gather: collect assigned token vectors
			batchData := make([]float32, nTok*hidden)
			for i, t := range tokens {
				copy(batchData[i*hidden:(i+1)*hidden], normed2Data[t*hidden:(t+1)*hidden])
			}
			batchGpu := ctx.NewTensor(batchData, nTok, hidden)

			// GPU SwiGLU: gate, silu, up, mul, down
			expert := block.moe.experts[eIdx]
			gate := gpuLinear(ctx, batchGpu, expert.wGate.weight)
			gateSilu := gpuSilu(ctx, gate)
			up := gpuLinear(ctx, batchGpu, expert.wUp.weight)
			fused := gpuMul(ctx, gateSilu, up)
			expertOut := gpuLinear(ctx, fused, expert.wDown.weight)
			eOutData := expertOut.Data()

			// Weighted scatter-add back to tokens
			for i, t := range tokens {
				k := expertWeightIdx[eIdx][i]
				alpha := wData[t*topK+k]
				for d := 0; d < hidden; d++ {
					moeOutData[t*hidden+d] += alpha * eOutData[i*hidden+d]
				}
			}

			// Release GPU buffers
			batchGpu.Release()
			gate.Release()
			gateSilu.Release()
			up.Release()
			fused.Release()
			expertOut.Release()
		}

		moeGpu := ctx.NewTensor(moeOutData, batchSeq, hidden)
		xNew2 := gpuAdd(ctx, x, moeGpu)
		x.Release()
		moeGpu.Release()
		normed2.Release()
		x = xNew2
	}

	// 3. Final RMSNorm
	finalEps := model.finalNorm.eps
	normedFinal := gpuRMSNorm(ctx, x, model.finalNorm.weight.Data(), batchSeq, hidden, finalEps)

	// 4. LM Head
	logits := gpuLinear(ctx, normedFinal, model.lmHead.weight)
	logitsData := logits.Data()

	// Cleanup
	x.Release()
	normedFinal.Release()
	logits.Release()

	// Return CPU tensor
	vocab := cfg.VocabSize
	return FromSlice(logitsData, NewShape(batch, seqLen, vocab))
}

// GpuTrainStep performs hybrid GPU forward + CPU backward training step.
// M1 unified memory (StorageModeShared) enables zero-copy CPU↔GPU access.
//
// Implementation note: Since backward requires cached intermediate activations from
// layer forward methods (lastInput, lastWeights, lastGate, etc.) and the current
// GpuForward implementation uses custom GPU kernels that bypass these caches,
// we run BOTH CPU forward (to populate caches) and GPU forward (to compute logits).
// This is suboptimal but demonstrates GPU forward correctness. Future optimization:
// integrate GPU kernels into layer forward methods to avoid duplicate computation.
func GpuTrainStep(ctx *MetalContext, trainer *Trainer, input, targets *Tensor) float32 {
	// CPU forward to populate caches for backward pass
	_ = trainer.model.Forward(input)

	// GPU forward to compute logits (validates GPU implementation)
	logitsGpu := GpuForward(ctx, trainer.model, input)

	// CPU backward + optimizer uses caches from CPU forward
	return trainer.TrainStepFromLogits(logitsGpu, targets)
}
