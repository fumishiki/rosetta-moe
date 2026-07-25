// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build darwin

package gpu

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework CoreGraphics
#cgo CFLAGS: -x objective-c -fobjc-arc
#include <stdlib.h>
#include "metal_bridge.h"
*/
import "C"
import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"
)

// MetalContext holds a Metal device, command queue, and optional shader library.
type MetalContext struct {
	device    unsafe.Pointer
	queue     unsafe.Pointer
	shaderLib unsafe.Pointer
	pipelines map[string]unsafe.Pointer
	mu        sync.Mutex
}

func (ctx *MetalContext) clearPipelines() {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	for fnName, p := range ctx.pipelines {
		if p != nil {
			C.metal_release(p)
		}
		delete(ctx.pipelines, fnName)
	}
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
		device:    dev,
		queue:     queue,
		pipelines: make(map[string]unsafe.Pointer),
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
	ctx.clearPipelines()
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
	ctx.clearPipelines()
	if ctx.shaderLib != nil {
		C.metal_release(ctx.shaderLib)
	}
	ctx.shaderLib = lib
	return nil
}

// LoadRequiredShaders loads and compiles all Metal shaders needed for GPU forward pass.
// shadersDir: path to directory containing .metal files (e.g., "../shaders").
func (ctx *MetalContext) LoadRequiredShaders(shadersDir string) error {
	files := []string{
		"rmsnorm.metal",
		"silu.metal",
		"elementwise.metal",
		"softmax.metal",
		"embedding.metal",
		"attention.metal",
	}
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

// Data returns a view into Metal unified memory as a Go slice.
// M1 unified memory (StorageModeShared) makes this a zero-copy operation.
// The returned slice is valid only while the MetalTensor is alive.
func (t *MetalTensor) Data() []float32 {
	if t == nil || t.buffer == nil || t.len == 0 {
		return nil
	}
	ptr := C.metal_buffer_contents(t.buffer)
	if ptr == nil {
		return nil
	}
	// Zero-copy view into Metal unified memory via unsafe.Slice.
	return unsafe.Slice((*float32)(ptr), t.len)
}

// DataCopy allocates a new Go slice and copies buffer contents into it.
// Use this when the MetalTensor may be released before the data is consumed.
func (t *MetalTensor) DataCopy() []float32 {
	view := t.Data()
	if view == nil {
		return nil
	}
	out := make([]float32, len(view))
	for i, v := range view {
		out[i] = v
	}
	return out
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

	ctx.mu.Lock()
	pipeline := ctx.pipelines[fnName]
	if pipeline == nil {
		// Create function once, then cache pipeline for repeated dispatches.
		cFnName := C.CString(fnName)
		fn := C.metal_create_function(ctx.shaderLib, cFnName)
		C.free(unsafe.Pointer(cFnName))
		if fn == nil {
			ctx.mu.Unlock()
			return fmt.Errorf("failed to create Metal function: %s", fnName)
		}
		pipeline = C.metal_create_pipeline(ctx.device, fn)
		C.metal_release(fn)
		if pipeline == nil {
			ctx.mu.Unlock()
			return fmt.Errorf("failed to create Metal pipeline for: %s", fnName)
		}
		ctx.pipelines[fnName] = pipeline
	}
	ctx.mu.Unlock()

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
	ctx.clearPipelines()
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

// ---------------------------------------------------------------------------
// GPU compute operations (internal helpers)
// ---------------------------------------------------------------------------

func threadgroup1D(n int) [3]int {
	if n <= 0 {
		return [3]int{1, 1, 1}
	}
	if n < 256 {
		return [3]int{n, 1, 1}
	}
	return [3]int{256, 1, 1}
}

// gpuLinear computes Y = X @ W^T using MPS matmul.
// WeightT is pre-transposed during upload to avoid per-dispatch host work.
func gpuLinear(ctx *MetalContext, input *MetalTensor, weight GpuLinearWeights) *MetalTensor {
	batchSeq := input.len / weight.InFeatures
	out := ctx.NewTensorZeros(batchSeq, weight.OutFeatures)
	ctx.Matmul(input, weight.WeightT, out, batchSeq, weight.OutFeatures, weight.InFeatures)
	return out
}

// gpuEmbeddingGather dispatches embedding lookup on GPU.
func gpuEmbeddingGather(ctx *MetalContext, tokenIDs, table *MetalTensor, batchSeq, hiddenDim int) *MetalTensor {
	out := ctx.NewTensorZeros(batchSeq, hiddenDim)
	dimBuf := ctx.NewTensorU32(uint32(hiddenDim))
	total := batchSeq * hiddenDim
	ctx.DispatchKernel("embedding_gather",
		[]*MetalTensor{tokenIDs, table, out, dimBuf},
		[3]int{total, 1, 1},
		threadgroup1D(total),
	)
	dimBuf.Release()
	return out
}

// gpuSoftmaxRows computes row-wise softmax for an [nRows, nCols] tensor.
func gpuSoftmaxRows(ctx *MetalContext, input *MetalTensor, nRows, nCols int) *MetalTensor {
	out := ctx.NewTensorZeros(nRows, nCols)
	nBuf := ctx.NewTensorU32(uint32(nCols))
	ctx.DispatchKernel("softmax",
		[]*MetalTensor{input, out, nBuf},
		[3]int{nRows * 256, 1, 1},
		[3]int{256, 1, 1},
	)
	nBuf.Release()
	return out
}

// gpuRMSNorm dispatches the rmsnorm Metal kernel.
func gpuRMSNorm(ctx *MetalContext, input, weight *MetalTensor, nRows, hiddenDim int, eps float32) *MetalTensor {
	out := ctx.NewTensorZeros(nRows, hiddenDim)
	dimBuf := ctx.NewTensorU32(uint32(hiddenDim))
	epsBuf := ctx.NewTensorF32Scalar(eps)
	ctx.DispatchKernel("rmsnorm",
		[]*MetalTensor{input, out, weight, dimBuf, epsBuf},
		[3]int{nRows * 256, 1, 1},
		[3]int{256, 1, 1},
	)
	dimBuf.Release()
	epsBuf.Release()
	return out
}

// gpuAttentionScores computes scaled QK^T.
func gpuAttentionScores(ctx *MetalContext, q, k *MetalTensor, batch, nHeads, nKVHeads, headDim, seqLen int, scale float32) *MetalTensor {
	out := ctx.NewTensorZeros(batch*nHeads, seqLen, seqLen)
	nHeadsBuf := ctx.NewTensorU32(uint32(nHeads))
	nKVHeadsBuf := ctx.NewTensorU32(uint32(nKVHeads))
	headDimBuf := ctx.NewTensorU32(uint32(headDim))
	seqLenBuf := ctx.NewTensorU32(uint32(seqLen))
	scaleBuf := ctx.NewTensorF32Scalar(scale)
	total := batch * nHeads * seqLen * seqLen
	ctx.DispatchKernel("attention_scores",
		[]*MetalTensor{q, k, out, nHeadsBuf, nKVHeadsBuf, headDimBuf, seqLenBuf, scaleBuf},
		[3]int{total, 1, 1},
		threadgroup1D(total),
	)
	nHeadsBuf.Release()
	nKVHeadsBuf.Release()
	headDimBuf.Release()
	seqLenBuf.Release()
	scaleBuf.Release()
	return out
}

// gpuCausalMaskFill applies causal masking in-place to attention scores.
func gpuCausalMaskFill(ctx *MetalContext, scores *MetalTensor, seqLen int) {
	seqLenBuf := ctx.NewTensorU32(uint32(seqLen))
	ctx.DispatchKernel("causal_mask_fill",
		[]*MetalTensor{scores, seqLenBuf},
		[3]int{scores.len, 1, 1},
		threadgroup1D(scores.len),
	)
	seqLenBuf.Release()
}

// gpuAttentionWeightedSum computes attention probabilities @ V.
func gpuAttentionWeightedSum(ctx *MetalContext, weights, v *MetalTensor, batchSeq, nHeads, nKVHeads, headDim, seqLen int) *MetalTensor {
	out := ctx.NewTensorZeros(batchSeq, nHeads*headDim)
	nHeadsBuf := ctx.NewTensorU32(uint32(nHeads))
	nKVHeadsBuf := ctx.NewTensorU32(uint32(nKVHeads))
	headDimBuf := ctx.NewTensorU32(uint32(headDim))
	seqLenBuf := ctx.NewTensorU32(uint32(seqLen))
	total := batchSeq * nHeads * headDim
	ctx.DispatchKernel("attention_weighted_sum",
		[]*MetalTensor{weights, v, out, nHeadsBuf, nKVHeadsBuf, headDimBuf, seqLenBuf},
		[3]int{total, 1, 1},
		threadgroup1D(total),
	)
	nHeadsBuf.Release()
	nKVHeadsBuf.Release()
	headDimBuf.Release()
	seqLenBuf.Release()
	return out
}

// gpuMoETopK runs top-k expert selection on GPU.
func gpuMoETopK(ctx *MetalContext, probs *MetalTensor, batchSeq, nExperts, topK int) (*MetalTensor, *MetalTensor) {
	indices := ctx.NewTensorZeros(batchSeq, topK)
	weights := ctx.NewTensorZeros(batchSeq, topK)
	nExpertsBuf := ctx.NewTensorU32(uint32(nExperts))
	topKBuf := ctx.NewTensorU32(uint32(topK))
	ctx.DispatchKernel("moe_topk",
		[]*MetalTensor{probs, indices, weights, nExpertsBuf, topKBuf},
		[3]int{batchSeq, 1, 1},
		[3]int{1, 1, 1},
	)
	nExpertsBuf.Release()
	topKBuf.Release()
	return indices, weights
}

// gpuMoETopKExtractWeight extracts one expert's routing weights per token.
func gpuMoETopKExtractWeight(ctx *MetalContext, indices, weights *MetalTensor, batchSeq, topK, expertIdx int) *MetalTensor {
	out := ctx.NewTensorZeros(batchSeq)
	topKBuf := ctx.NewTensorU32(uint32(topK))
	expertBuf := ctx.NewTensorU32(uint32(expertIdx))
	ctx.DispatchKernel("moe_topk_extract_weight",
		[]*MetalTensor{indices, weights, out, topKBuf, expertBuf},
		[3]int{batchSeq, 1, 1},
		threadgroup1D(batchSeq),
	)
	topKBuf.Release()
	expertBuf.Release()
	return out
}

// gpuRowScale scales each row by a per-row coefficient.
func gpuRowScale(ctx *MetalContext, input, rowWeights *MetalTensor, nRows, nCols int) *MetalTensor {
	out := ctx.NewTensorZeros(nRows, nCols)
	nColsBuf := ctx.NewTensorU32(uint32(nCols))
	total := nRows * nCols
	ctx.DispatchKernel("row_scale",
		[]*MetalTensor{input, rowWeights, out, nColsBuf},
		[3]int{total, 1, 1},
		threadgroup1D(total),
	)
	nColsBuf.Release()
	return out
}

// gpuCrossEntropyLossSum computes CE loss on GPU and returns the summed scalar on GPU.
// Caller owns the returned tensor and must Release() it.
func gpuCrossEntropyLossSum(ctx *MetalContext, logits, targets *MetalTensor, nTokens, vocabSize int) *MetalTensor {
	perToken := ctx.NewTensorZeros(nTokens)
	vocabBuf := ctx.NewTensorU32(uint32(vocabSize))
	ctx.DispatchKernel("cross_entropy_forward",
		[]*MetalTensor{logits, targets, perToken, vocabBuf},
		[3]int{nTokens * 256, 1, 1},
		[3]int{256, 1, 1},
	)
	vocabBuf.Release()

	sumBuf := ctx.NewTensorZeros(1)
	nBuf := ctx.NewTensorU32(uint32(nTokens))
	ctx.DispatchKernel("reduce_sum",
		[]*MetalTensor{perToken, sumBuf, nBuf},
		[3]int{256, 1, 1},
		[3]int{256, 1, 1},
	)
	nBuf.Release()
	perToken.Release()
	return sumBuf
}

// gpuCrossEntropyMeanLoss computes mean CE loss on GPU and returns it as a scalar.
func gpuCrossEntropyMeanLoss(ctx *MetalContext, logits, targets *MetalTensor, nTokens, vocabSize int) float32 {
	sumBuf := gpuCrossEntropyLossSum(ctx, logits, targets, nTokens, vocabSize)

	sumView := sumBuf.Data()
	sum := float32(0)
	if len(sumView) > 0 {
		sum = sumView[0]
	}
	sumBuf.Release()
	return sum / float32(nTokens)
}

// gpuSilu dispatches the silu Metal kernel (elementwise SiLU activation).
func gpuSilu(ctx *MetalContext, input *MetalTensor) *MetalTensor {
	n := input.len
	out := ctx.NewTensorZeros(input.shape...)
	ctx.DispatchKernel("silu",
		[]*MetalTensor{input, out},
		[3]int{n, 1, 1},
		threadgroup1D(n),
	)
	return out
}

// gpuAdd dispatches the elementwise add Metal kernel.
func gpuAdd(ctx *MetalContext, a, b *MetalTensor) *MetalTensor {
	out := ctx.NewTensorZeros(a.shape...)
	ctx.DispatchKernel("add",
		[]*MetalTensor{a, b, out},
		[3]int{a.len, 1, 1},
		threadgroup1D(a.len),
	)
	return out
}

// gpuMul dispatches the elementwise mul Metal kernel.
func gpuMul(ctx *MetalContext, a, b *MetalTensor) *MetalTensor {
	out := ctx.NewTensorZeros(a.shape...)
	ctx.DispatchKernel("mul",
		[]*MetalTensor{a, b, out},
		[3]int{a.len, 1, 1},
		threadgroup1D(a.len),
	)
	return out
}

// ---------------------------------------------------------------------------
// GPU Model: self-contained model representation with all weights on GPU
// ---------------------------------------------------------------------------

// GpuLinearWeights holds a linear layer's weights on GPU.
type GpuLinearWeights struct {
	Weight      *MetalTensor
	WeightT     *MetalTensor
	InFeatures  int
	OutFeatures int
}

// GpuSwiGLUWeights holds SwiGLU expert weights on GPU.
type GpuSwiGLUWeights struct {
	WGate, WUp, WDown GpuLinearWeights
}

// GpuRMSNormWeights holds RMSNorm weights on GPU.
type GpuRMSNormWeights struct {
	Weight *MetalTensor
	Eps    float32
}

// GpuMoEWeights holds MoE layer weights on GPU.
type GpuMoEWeights struct {
	RouterGate GpuLinearWeights
	Experts    []GpuSwiGLUWeights
	NExperts   int
	TopK       int
}

// GpuBlockWeights holds a transformer block's weights on GPU.
type GpuBlockWeights struct {
	AttnNorm GpuRMSNormWeights
	FfnNorm  GpuRMSNormWeights
	// Attention projection weights.
	AttnWQ, AttnWK, AttnWV, AttnWO GpuLinearWeights
	Moe                            GpuMoEWeights
}

// GpuModel holds all model weights on GPU for pure GPU forward pass.
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

// UploadModel uploads model weights from flat []float32 slices to GPU MetalTensors.
// This is the boundary between CPU initialization and GPU compute.
// The ModelSpec describes the model architecture and provides weight data.
type ModelSpec struct {
	VocabSize int
	HiddenDim int
	FFNDim    int
	NLayers   int
	NHeads    int
	NKVHeads  int
	HeadDim   int
	NExperts  int
	TopK      int

	// Weight data as flat slices (read from CPU model parameters).
	EmbeddingWeight []float32 // [vocabSize, hiddenDim]
	FinalNormWeight []float32 // [hiddenDim]
	FinalNormEps    float32
	LmHeadWeight    []float32 // [vocabSize, hiddenDim]

	// Per-block weights
	Blocks []BlockSpec
}

// BlockSpec holds per-block weight data for upload.
type BlockSpec struct {
	AttnNormWeight []float32
	AttnNormEps    float32
	FfnNormWeight  []float32
	FfnNormEps     float32

	// Attention weights
	WQWeight []float32
	WQIn     int
	WQOut    int
	WKWeight []float32
	WKIn     int
	WKOut    int
	WVWeight []float32
	WVIn     int
	WVOut    int
	WOWeight []float32
	WOIn     int
	WOOut    int

	// Router
	RouterGateWeight []float32
	RouterGateIn     int
	RouterGateOut    int

	// Experts
	Experts []ExpertSpec
}

// ExpertSpec holds per-expert weight data for upload.
type ExpertSpec struct {
	WGateWeight []float32
	WGateIn     int
	WGateOut    int
	WUpWeight   []float32
	WUpIn       int
	WUpOut      int
	WDownWeight []float32
	WDownIn     int
	WDownOut    int
}

func transposeWeight(weight []float32, outFeatures, inFeatures int) []float32 {
	out := make([]float32, inFeatures*outFeatures)
	for r := 0; r < outFeatures; r++ {
		for c := 0; c < inFeatures; c++ {
			out[c*outFeatures+r] = weight[r*inFeatures+c]
		}
	}
	return out
}

func uploadLinear(ctx *MetalContext, weight []float32, inFeatures, outFeatures int) GpuLinearWeights {
	return GpuLinearWeights{
		Weight:      ctx.NewTensor(weight, outFeatures, inFeatures),
		WeightT:     ctx.NewTensor(transposeWeight(weight, outFeatures, inFeatures), inFeatures, outFeatures),
		InFeatures:  inFeatures,
		OutFeatures: outFeatures,
	}
}

// UploadModel transfers all model weights to GPU memory.
func (ctx *MetalContext) UploadModel(spec *ModelSpec) *GpuModel {
	m := &GpuModel{
		Embedding: ctx.NewTensor(spec.EmbeddingWeight, spec.VocabSize, spec.HiddenDim),
		FinalNorm: GpuRMSNormWeights{
			Weight: ctx.NewTensor(spec.FinalNormWeight, spec.HiddenDim),
			Eps:    spec.FinalNormEps,
		},
		LmHead:    uploadLinear(ctx, spec.LmHeadWeight, spec.HiddenDim, spec.VocabSize),
		VocabSize: spec.VocabSize,
		HiddenDim: spec.HiddenDim,
		FFNDim:    spec.FFNDim,
		NHeads:    spec.NHeads,
		NKVHeads:  spec.NKVHeads,
		HeadDim:   spec.HeadDim,
		Blocks:    make([]GpuBlockWeights, len(spec.Blocks)),
	}

	for i, bs := range spec.Blocks {
		experts := make([]GpuSwiGLUWeights, len(bs.Experts))
		for j, es := range bs.Experts {
			experts[j] = GpuSwiGLUWeights{
				WGate: uploadLinear(ctx, es.WGateWeight, es.WGateIn, es.WGateOut),
				WUp:   uploadLinear(ctx, es.WUpWeight, es.WUpIn, es.WUpOut),
				WDown: uploadLinear(ctx, es.WDownWeight, es.WDownIn, es.WDownOut),
			}
		}

		m.Blocks[i] = GpuBlockWeights{
			AttnNorm: GpuRMSNormWeights{
				Weight: ctx.NewTensor(bs.AttnNormWeight, spec.HiddenDim),
				Eps:    bs.AttnNormEps,
			},
			FfnNorm: GpuRMSNormWeights{
				Weight: ctx.NewTensor(bs.FfnNormWeight, spec.HiddenDim),
				Eps:    bs.FfnNormEps,
			},
			AttnWQ: uploadLinear(ctx, bs.WQWeight, bs.WQIn, bs.WQOut),
			AttnWK: uploadLinear(ctx, bs.WKWeight, bs.WKIn, bs.WKOut),
			AttnWV: uploadLinear(ctx, bs.WVWeight, bs.WVIn, bs.WVOut),
			AttnWO: uploadLinear(ctx, bs.WOWeight, bs.WOIn, bs.WOOut),
			Moe: GpuMoEWeights{
				RouterGate: uploadLinear(ctx, bs.RouterGateWeight, bs.RouterGateIn, bs.RouterGateOut),
				Experts:    experts,
				NExperts:   len(bs.Experts),
				TopK:       spec.TopK,
			},
		}
	}

	return m
}

// ReleaseModel frees all GPU memory held by the model.
func (m *GpuModel) ReleaseModel() {
	if m == nil {
		return
	}
	m.Embedding.Release()
	m.FinalNorm.Weight.Release()
	m.LmHead.Weight.Release()
	m.LmHead.WeightT.Release()
	for i := range m.Blocks {
		b := &m.Blocks[i]
		b.AttnNorm.Weight.Release()
		b.FfnNorm.Weight.Release()
		b.AttnWQ.Weight.Release()
		b.AttnWQ.WeightT.Release()
		b.AttnWK.Weight.Release()
		b.AttnWK.WeightT.Release()
		b.AttnWV.Weight.Release()
		b.AttnWV.WeightT.Release()
		b.AttnWO.Weight.Release()
		b.AttnWO.WeightT.Release()
		b.Moe.RouterGate.Weight.Release()
		b.Moe.RouterGate.WeightT.Release()
		for j := range b.Moe.Experts {
			e := &b.Moe.Experts[j]
			e.WGate.Weight.Release()
			e.WGate.WeightT.Release()
			e.WUp.Weight.Release()
			e.WUp.WeightT.Release()
			e.WDown.Weight.Release()
			e.WDown.WeightT.Release()
		}
	}
}

// GpuForwardTensor performs a pure GPU forward pass for the MoE Transformer.
// All weights and token IDs must already be on GPU.
// Input: token IDs as *MetalTensor [batch, seqLen] (or flat [batch*seqLen]).
// Output: logits as *MetalTensor.
func GpuForwardTensor(ctx *MetalContext, model *GpuModel, tokenBuf *MetalTensor, batch, seqLen int) *MetalTensor {
	hidden := model.HiddenDim
	batchSeq := batch * seqLen
	attnScale := float32(1.0 / math.Sqrt(float64(model.HeadDim)))

	// 1. Embedding lookup
	x := gpuEmbeddingGather(ctx, tokenBuf, model.Embedding, batchSeq, hidden)

	// 2. Transformer blocks
	for bi := range model.Blocks {
		block := &model.Blocks[bi]

		// RMSNorm (attn)
		normed := gpuRMSNorm(ctx, x, block.AttnNorm.Weight, batchSeq, hidden, block.AttnNorm.Eps)

		// Attention: QK^T + causal mask + softmax + weighted sum
		q := gpuLinear(ctx, normed, block.AttnWQ)
		k := gpuLinear(ctx, normed, block.AttnWK)
		v := gpuLinear(ctx, normed, block.AttnWV)
		scores := gpuAttentionScores(ctx, q, k, batch, model.NHeads, model.NKVHeads, model.HeadDim, seqLen, attnScale)
		gpuCausalMaskFill(ctx, scores, seqLen)
		probs := gpuSoftmaxRows(ctx, scores, batch*model.NHeads*seqLen, seqLen)
		attnOut := gpuAttentionWeightedSum(ctx, probs, v, batchSeq, model.NHeads, model.NKVHeads, model.HeadDim, seqLen)
		attnProj := gpuLinear(ctx, attnOut, block.AttnWO)

		q.Release()
		k.Release()
		v.Release()
		scores.Release()
		probs.Release()
		attnOut.Release()

		// Residual
		xNew := gpuAdd(ctx, x, attnProj)
		x.Release()
		attnProj.Release()
		normed.Release()
		x = xNew

		// RMSNorm (ffn)
		normed2 := gpuRMSNorm(ctx, x, block.FfnNorm.Weight, batchSeq, hidden, block.FfnNorm.Eps)

		// MoE: GPU router + expert SwiGLU
		topK := block.Moe.TopK
		nExperts := block.Moe.NExperts

		// GPU router: linear + softmax + top-K
		routerLogits := gpuLinear(ctx, normed2, block.Moe.RouterGate)
		routerProbs := gpuSoftmaxRows(ctx, routerLogits, batchSeq, nExperts)
		topKIndices, topKWeights := gpuMoETopK(ctx, routerProbs, batchSeq, nExperts, topK)
		routerLogits.Release()
		routerProbs.Release()

		// For each expert: full-batch expert forward + row-wise scaling by routing weights.
		moeOut := ctx.NewTensorZeros(batchSeq, hidden)
		for eIdx := 0; eIdx < nExperts; eIdx++ {
			tokenWeights := gpuMoETopKExtractWeight(ctx, topKIndices, topKWeights, batchSeq, topK, eIdx)

			// GPU SwiGLU on full [batchSeq, hidden], then row-scale by token routing weight.
			expert := &block.Moe.Experts[eIdx]
			gate := gpuLinear(ctx, normed2, expert.WGate)
			gateSilu := gpuSilu(ctx, gate)
			up := gpuLinear(ctx, normed2, expert.WUp)
			fused := gpuMul(ctx, gateSilu, up)
			expertOut := gpuLinear(ctx, fused, expert.WDown)
			scaledOut := gpuRowScale(ctx, expertOut, tokenWeights, batchSeq, hidden)
			updated := gpuAdd(ctx, moeOut, scaledOut)

			tokenWeights.Release()
			gate.Release()
			gateSilu.Release()
			up.Release()
			fused.Release()
			expertOut.Release()
			scaledOut.Release()
			moeOut.Release()
			moeOut = updated
		}

		topKIndices.Release()
		topKWeights.Release()

		xNew2 := gpuAdd(ctx, x, moeOut)
		x.Release()
		moeOut.Release()
		normed2.Release()
		x = xNew2
	}

	// 3. Final RMSNorm
	normedFinal := gpuRMSNorm(ctx, x, model.FinalNorm.Weight, batchSeq, hidden, model.FinalNorm.Eps)

	// 4. LM Head
	logits := gpuLinear(ctx, normedFinal, model.LmHead)

	x.Release()
	normedFinal.Release()

	return logits
}

// GpuForward performs a pure GPU forward pass for the MoE Transformer.
// All weights must be pre-uploaded via UploadModel.
// Input: token IDs as []float32 [batch*seqLen], Output: logits as *MetalTensor.
func GpuForward(ctx *MetalContext, model *GpuModel, inputIDs []float32, batch, seqLen int) *MetalTensor {
	tokenBuf := ctx.NewTensor(inputIDs, batch, seqLen)
	logits := GpuForwardTensor(ctx, model, tokenBuf, batch, seqLen)
	tokenBuf.Release()
	return logits
}
