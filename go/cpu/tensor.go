// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

// Package cpu implements a Mixture-of-Experts Transformer from scratch in Go.
//
// All tensor storage uses flat []float32 slices in row-major order.
// Matrix multiplication is delegated to Apple Accelerate (cblas_sgemm) via CGO.
// No external Go dependencies beyond the standard library.
package cpu

import (
	"fmt"
	"math"
	"strings"
)

// Global LCG state for reproducible weight initialization (matches Rust/Python/Julia).
var lcgState uint64 = 42

// SeedRNG sets the global LCG state for reproducible weight initialization.
func SeedRNG(seed uint64) {
	lcgState = seed
	lcgNormalHasCache = false
}

// lcgUniform returns a uniform random float64 in [1e-10, 1.0] using Knuth's MMIX LCG.
func lcgUniform() float64 {
	lcgState = lcgState*6364136223846793005 + 1
	u := float64(lcgState) / float64(^uint64(0))
	if u < 1e-10 {
		u = 1e-10
	}
	return u
}

var lcgNormalCache float64
var lcgNormalHasCache bool

func lcgNormal() float64 {
	if lcgNormalHasCache {
		lcgNormalHasCache = false
		return lcgNormalCache
	}
	u1 := lcgUniform()
	u2 := lcgUniform()
	r := math.Sqrt(-2.0 * math.Log(u1))
	theta := 2.0 * math.Pi * u2
	lcgNormalHasCache = true
	lcgNormalCache = r * math.Sin(theta)
	return r * math.Cos(theta)
}

// DType enumerates supported data types.
type DType uint8

const (
	F32 DType = iota
	F16
	BF16
	I32
	I64
)

// Size returns the byte width of the data type.
func (d DType) Size() int {
	switch d {
	case F32, I32:
		return 4
	case F16, BF16:
		return 2
	case I64:
		return 8
	default:
		return 4
	}
}

// String returns a human-readable name for the data type.
func (d DType) String() string {
	names := [...]string{"f32", "f16", "bf16", "i32", "i64"}
	if int(d) < len(names) {
		return names[d]
	}
	return "unknown"
}

// Shape represents the dimensions of a tensor.
type Shape struct{ dims []int }

// NewShape creates a Shape from variadic dimension sizes.
func NewShape(dims ...int) Shape {
	d := make([]int, len(dims))
	copy(d, dims)
	return Shape{dims: d}
}

// Dims returns a copy of the dimension sizes.
func (s Shape) Dims() []int {
	d := make([]int, len(s.dims))
	copy(d, s.dims)
	return d
}

// DimsRef returns a direct reference to the internal dimension slice.
func (s Shape) DimsRef() []int {
	return s.dims
}

// NDim returns the number of dimensions.
func (s Shape) NDim() int { return len(s.dims) }

// Numel returns the total number of elements (product of all dimensions).
func (s Shape) Numel() int {
	if len(s.dims) == 0 {
		return 0
	}
	return prod(s.dims)
}

// At returns the size of dimension dim. Negative indices count from the end.
func (s Shape) At(dim int) int {
	if dim < 0 {
		dim += len(s.dims)
	}
	if dim < 0 || dim >= len(s.dims) {
		return 0
	}
	return s.dims[dim]
}

// Strides returns row-major strides for the shape.
func (s Shape) Strides() []int {
	if len(s.dims) == 0 {
		return nil
	}
	strides := make([]int, len(s.dims))
	strides[len(s.dims)-1] = 1
	for i := len(s.dims) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * s.dims[i+1]
	}
	return strides
}

// Equal returns true if two shapes have identical dimensions.
func (s Shape) Equal(other Shape) bool {
	if len(s.dims) != len(other.dims) {
		return false
	}
	for i := range s.dims {
		if s.dims[i] != other.dims[i] {
			return false
		}
	}
	return true
}

// String formats the shape as "[d0, d1, ...]".
func (s Shape) String() string {
	parts := make([]string, len(s.dims))
	for i, d := range s.dims {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// Broadcast computes the broadcast-compatible output shape for two inputs.
func Broadcast(a, b Shape) (Shape, error) {
	maxLen := len(a.dims)
	if len(b.dims) > maxLen {
		maxLen = len(b.dims)
	}
	result := make([]int, maxLen)
	for i := range result {
		dimA, dimB := 1, 1
		if i < len(a.dims) {
			dimA = a.dims[len(a.dims)-1-i]
		}
		if i < len(b.dims) {
			dimB = b.dims[len(b.dims)-1-i]
		}
		if dimA != dimB && dimA != 1 && dimB != 1 {
			return Shape{}, fmt.Errorf("cannot broadcast shapes %v and %v", a, b)
		}
		if dimA > dimB {
			result[maxLen-1-i] = dimA
		} else {
			result[maxLen-1-i] = dimB
		}
	}
	return Shape{dims: result}, nil
}

// NegInf is the most negative finite float32, used as -infinity for masking.
const NegInf = -float32(math.MaxFloat32)

// ---------------------------------------------------------------------------
// Pure-float32 math functions
// ---------------------------------------------------------------------------

// ExpF32 computes exp(x) in pure float32.
func ExpF32(x float32) float32 {
	if x > 88.72 {
		return float32(math.Inf(1))
	}
	if x < -87.33 {
		return 0
	}
	const (
		invLn2 = float32(1.4426950)
		ln2Hi  = float32(0.6931458)
		ln2Lo  = float32(1.4286068e-06)
	)
	var k int32
	if x >= 0 {
		k = int32(x*invLn2 + 0.5)
	} else {
		k = int32(x*invLn2 - 0.5)
	}
	kf := float32(k)
	r := x - kf*ln2Hi - kf*ln2Lo
	r2 := r * r
	p := float32(1) + r + r2*(0.5+r*(0.16666667+r*(0.04166668+r*0.008333334)))
	return p * math.Float32frombits(uint32(127+k)<<23)
}

// SqrtF32 computes sqrt(x) via the fast inverse square root trick.
func SqrtF32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	bits := math.Float32bits(x)
	bits = 0x5f3759df - (bits >> 1)
	y := math.Float32frombits(bits)
	half := 0.5 * x
	y = y * (1.5 - half*y*y)
	y = y * (1.5 - half*y*y)
	return x * y
}

// LogF32 computes ln(x) in pure float32.
func LogF32(x float32) float32 {
	if x <= 0 {
		return NegInf
	}
	bits := math.Float32bits(x)
	e := int32((bits>>23)&0xFF) - 127
	bits = (bits & 0x007FFFFF) | 0x3F800000
	m := math.Float32frombits(bits)
	s := (m - 1) / (m + 1)
	s2 := s * s
	p := 2.0 * s * (1 + s2*(0.33333334+s2*(0.2+s2*0.14285715)))
	return float32(e)*0.6931472 + p
}

// PowF32 computes base^exp in float32 via exp(exp * ln(base)).
func PowF32(base, exp float32) float32 {
	if base <= 0 {
		return 0
	}
	return ExpF32(exp * LogF32(base))
}

// SinF32 computes sin(x) via range reduction.
func SinF32(x float32) float32 {
	const (
		twoPi  = float32(6.2831855)
		pi     = float32(3.1415927)
		halfPi = float32(1.5707964)
	)
	x -= float32(int32(x/twoPi)) * twoPi
	if x < 0 {
		x += twoPi
	}
	sign := float32(1)
	if x > pi {
		sign = -1
		x -= pi
	}
	if x > halfPi {
		x = pi - x
	}
	x2 := x * x
	return sign * x * (1 - x2*(0.16666667-x2*(0.008333334-x2*0.00019841270)))
}

// CosF32 computes cos(x) = sin(x + pi/2).
func CosF32(x float32) float32 { return SinF32(x + 1.5707964) }

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

// Tensor stores multi-dimensional float32 data in a contiguous flat slice.
type Tensor struct {
	data  []float32
	shape Shape
	dtype DType
	Grad  []float32
}

// ZeroGrad resets the gradient.
func (t *Tensor) ZeroGrad() {
	n := len(t.data)
	if t.Grad != nil && len(t.Grad) == n {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	} else {
		t.Grad = nil
	}
}

// AccumulateGrad adds grad element-wise into t.Grad, allocating if nil.
func (t *Tensor) AccumulateGrad(grad []float32) {
	if t.Grad == nil {
		t.Grad = make([]float32, len(t.data))
	}
	for i, g := range grad {
		t.Grad[i] += g
	}
}

// New allocates a zero-filled tensor of the given shape and dtype.
func New(shape Shape, dtype DType) *Tensor {
	return &Tensor{data: make([]float32, shape.Numel()), shape: shape, dtype: dtype}
}

// Zeros is an alias for New (zero-filled tensor).
func Zeros(shape Shape, dtype DType) *Tensor { return New(shape, dtype) }

// Ones allocates a tensor filled with 1.0.
func Ones(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = 1
	}
	return t
}

// FromSlice creates a tensor by copying the provided data.
func FromSlice(data []float32, shape Shape) *Tensor {
	if len(data) != shape.Numel() {
		panic(fmt.Sprintf("data length %d != shape numel %d", len(data), shape.Numel()))
	}
	d := make([]float32, len(data))
	copy(d, data)
	return &Tensor{data: d, shape: shape, dtype: F32}
}

// FromSliceNoCopy creates a tensor that directly owns the provided slice (no copy).
func FromSliceNoCopy(data []float32, shape Shape) *Tensor {
	if len(data) != shape.Numel() {
		panic(fmt.Sprintf("data length %d != shape numel %d", len(data), shape.Numel()))
	}
	return &Tensor{data: data, shape: shape, dtype: F32}
}

// Randn allocates a tensor filled with standard normal random values.
func Randn(shape Shape, dtype DType) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(lcgNormal())
	}
	return t
}

// RandnWithStd allocates a tensor filled with normal random values scaled by std.
func RandnWithStd(shape Shape, dtype DType, std float32) *Tensor {
	t := New(shape, dtype)
	for i := range t.data {
		t.data[i] = float32(lcgNormal()) * std
	}
	return t
}

// Shape returns the tensor's shape.
func (t *Tensor) Shape() Shape { return t.shape }

// DType returns the tensor's data type tag.
func (t *Tensor) DType() DType { return t.dtype }

// DataPtr returns the underlying storage slice directly (no copy).
func (t *Tensor) DataPtr() []float32 { return t.data }

// Data returns a copy of the underlying storage.
func (t *Tensor) Data() []float32 {
	d := make([]float32, len(t.data))
	copy(d, t.data)
	return d
}

func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != t.shape.NDim() {
		panic(fmt.Sprintf("expected %d indices, got %d", t.shape.NDim(), len(indices)))
	}
	idx := 0
	strides := t.shape.Strides()
	for i, index := range indices {
		if index < 0 || index >= t.shape.At(i) {
			panic(fmt.Sprintf("index %d out of bounds for dim %d with size %d", index, i, t.shape.At(i)))
		}
		idx += index * strides[i]
	}
	return idx
}

// At reads a single element by multi-dimensional index.
func (t *Tensor) At(indices ...int) float32 { return t.data[t.flatIndex(indices)] }

// Set writes a single element by multi-dimensional index.
func (t *Tensor) Set(value float32, indices ...int) { t.data[t.flatIndex(indices)] = value }

// Clone returns a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor { return FromSlice(t.data, t.shape) }

// Reshape returns a new tensor sharing the same backing data but with a different shape.
func (t *Tensor) Reshape(s Shape) *Tensor {
	if t.shape.Numel() != s.Numel() {
		panic(fmt.Sprintf("cannot reshape %v to %v: different numel", t.shape, s))
	}
	return &Tensor{data: t.data, shape: s, dtype: t.dtype}
}

func (t *Tensor) assertShape(other *Tensor) {
	if !t.shape.Equal(other.shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", t.shape, other.shape))
	}
}

// Add returns element-wise t + o.
func (t *Tensor) Add(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
	return r
}

// Sub returns element-wise t - o.
func (t *Tensor) Sub(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
	return r
}

// Mul returns element-wise t * o (Hadamard product).
func (t *Tensor) Mul(o *Tensor) *Tensor {
	t.assertShape(o)
	r := New(t.shape, t.dtype)
	a, b, dst := t.data, o.data, r.data
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
	return r
}

// Scale returns t * s (scalar multiplication).
func (t *Tensor) Scale(s float32) *Tensor {
	r := New(t.shape, t.dtype)
	src, dst := t.data, r.data
	for i := range dst {
		dst[i] = src[i] * s
	}
	return r
}

// SiLU returns the SiLU (Swish) activation applied element-wise.
func (t *Tensor) SiLU() *Tensor {
	r := New(t.shape, t.dtype)
	src, dst := t.data, r.data
	for i, x := range src {
		dst[i] = x / (1 + ExpF32(-x))
	}
	return r
}

// AddInPlace adds other to t element-wise, mutating t.
func (t *Tensor) AddInPlace(other *Tensor) {
	t.assertShape(other)
	a, b := t.data, other.data
	for i := range a {
		a[i] += b[i]
	}
}

// SiLUInPlace applies SiLU activation in-place.
func (t *Tensor) SiLUInPlace() {
	for i, x := range t.data {
		t.data[i] = x / (1 + ExpF32(-x))
	}
}

// MulInPlace multiplies t by other element-wise, mutating t.
func (t *Tensor) MulInPlace(other *Tensor) {
	t.assertShape(other)
	a, b := t.data, other.data
	for i := range a {
		a[i] *= b[i]
	}
}

// ScaleInPlace multiplies every element of t by s, mutating t.
func (t *Tensor) ScaleInPlace(s float32) {
	for i := range t.data {
		t.data[i] *= s
	}
}

func softmaxCore(src, dst []float32, lastDim, numVectors int) {
	for v := 0; v < numVectors; v++ {
		off := v * lastDim
		sRow := src[off : off+lastDim]
		dRow := dst[off : off+lastDim]

		maxVal := sRow[0]
		for i := 1; i < lastDim; i++ {
			if sRow[i] > maxVal {
				maxVal = sRow[i]
			}
		}
		sum := float32(0)
		for i := 0; i < lastDim; i++ {
			e := ExpF32(sRow[i] - maxVal)
			dRow[i] = e
			sum += e
		}
		invSum := 1.0 / sum
		for i := 0; i < lastDim; i++ {
			dRow[i] *= invSum
		}
	}
}

// Softmax computes row-wise softmax along the last dimension.
func (t *Tensor) Softmax() *Tensor {
	if t.shape.NDim() < 1 {
		panic("softmax requires at least 1 dimension")
	}
	result := New(t.shape, t.dtype)
	lastDim := t.shape.At(-1)
	numVectors := t.shape.Numel() / lastDim
	softmaxCore(t.data, result.data, lastDim, numVectors)
	return result
}

// SoftmaxInto computes row-wise softmax into a pre-allocated output tensor.
func (t *Tensor) SoftmaxInto(out *Tensor) {
	t.assertShape(out)
	lastDim := t.shape.At(-1)
	numVectors := t.shape.Numel() / lastDim
	softmaxCore(t.data, out.data, lastDim, numVectors)
}

// Matmul computes matrix multiplication C = A @ B.
func Matmul(a, b *Tensor) *Tensor {
	if a.shape.NDim() < 2 || b.shape.NDim() < 2 {
		panic("matmul requires at least 2D tensors")
	}
	aM, aK := a.shape.At(-2), a.shape.At(-1)
	bK, bN := b.shape.At(-2), b.shape.At(-1)
	if aK != bK {
		panic(fmt.Sprintf("matmul dimension mismatch: %d vs %d", aK, bK))
	}

	var batchSize int
	var resultShape Shape
	switch {
	case a.shape.NDim() == 2 && b.shape.NDim() == 2:
		batchSize = 1
		resultShape = NewShape(aM, bN)
	case a.shape.NDim() == 3 && b.shape.NDim() == 3:
		if a.shape.At(0) != b.shape.At(0) {
			panic(fmt.Sprintf("matmul batch mismatch: %d vs %d", a.shape.At(0), b.shape.At(0)))
		}
		batchSize = a.shape.At(0)
		resultShape = NewShape(batchSize, aM, bN)
	default:
		panic("unsupported batch dimensions")
	}

	result := New(resultShape, a.dtype)
	aStride, bStride, cStride := aM*aK, bK*bN, aM*bN

	for batch := 0; batch < batchSize; batch++ {
		aOff, bOff, cOff := batch*aStride, batch*bStride, batch*cStride
		Sgemm(aM, bN, aK,
			1.0, a.data[aOff:aOff+aStride], aK,
			b.data[bOff:bOff+bStride], bN,
			0.0, result.data[cOff:cOff+cStride], bN)
	}
	return result
}

// MatmulTransposedB computes C = A @ B^T without materializing the transpose.
func MatmulTransposedB(a, b *Tensor) *Tensor {
	if a.shape.NDim() != 2 || b.shape.NDim() != 2 {
		panic("MatmulTransposedB requires 2D tensors")
	}
	aM, aK := a.shape.At(-2), a.shape.At(-1)
	bN, bK := b.shape.At(-2), b.shape.At(-1)
	if aK != bK {
		panic(fmt.Sprintf("matmulT dimension mismatch: %d vs %d", aK, bK))
	}
	result := New(NewShape(aM, bN), a.dtype)
	SgemmTransB(aM, bN, aK,
		1.0, a.data, aK,
		b.data, bK,
		0.0, result.data, bN)
	return result
}

// Transpose swaps the last two dimensions.
func (t *Tensor) Transpose() *Tensor {
	if t.shape.NDim() < 2 {
		panic("transpose requires at least 2D tensor")
	}
	dims := t.shape.Dims()
	dims[len(dims)-1], dims[len(dims)-2] = dims[len(dims)-2], dims[len(dims)-1]
	result := New(NewShape(dims...), t.dtype)
	rows, cols := t.shape.At(-2), t.shape.At(-1)
	batchSize := t.shape.Numel() / (rows * cols)
	for batch := 0; batch < batchSize; batch++ {
		srcOff, dstOff := batch*rows*cols, batch*cols*rows
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				result.data[dstOff+j*rows+i] = t.data[srcOff+i*cols+j]
			}
		}
	}
	return result
}

// Sum returns the sum of all elements.
func (t *Tensor) Sum() float32 {
	sum := float32(0)
	for _, v := range t.data {
		sum += v
	}
	return sum
}

// Mean returns the arithmetic mean of all elements.
func (t *Tensor) Mean() float32 { return t.Sum() / float32(len(t.data)) }

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

func prod(xs []int) int {
	n := 1
	for _, x := range xs {
		n *= x
	}
	return n
}

func splitLast(dims []int) (leading []int, leadingSize int, last int) {
	if len(dims) == 0 {
		panic("shape must have at least one dimension")
	}
	last = dims[len(dims)-1]
	leading = dims[:len(dims)-1]
	leadingSize = prod(leading)
	return leading, leadingSize, last
}

func withLastDim(dims []int, last int) Shape {
	out := append(append([]int(nil), dims...), last)
	return NewShape(out...)
}

func concatParams(groups ...[]*Tensor) []*Tensor {
	total := 0
	for _, g := range groups {
		total += len(g)
	}
	out := make([]*Tensor, 0, total)
	for _, g := range groups {
		out = append(out, g...)
	}
	return out
}

func cloneInts(src []int) []int {
	dst := make([]int, len(src))
	copy(dst, src)
	return dst
}

func argmax(xs []float32) (int, float32) {
	bestIdx, bestVal := 0, xs[0]
	for i := 1; i < len(xs); i++ {
		if xs[i] > bestVal {
			bestIdx, bestVal = i, xs[i]
		}
	}
	return bestIdx, bestVal
}

func normalizeInPlace(xs []float32) {
	sum := float32(0)
	for _, v := range xs {
		sum += v
	}
	if sum == 0 {
		return
	}
	invSum := 1.0 / sum
	for i := range xs {
		xs[i] *= invSum
	}
}

func resetBools(xs []bool) {
	for i := range xs {
		xs[i] = false
	}
}
