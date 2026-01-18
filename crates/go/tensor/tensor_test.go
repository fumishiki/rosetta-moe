package tensor

import (
	"math"
	"testing"
)

func TestShape(t *testing.T) {
	s := NewShape(2, 3, 4)
	if s.NDim() != 3 {
		t.Errorf("expected 3 dims, got %d", s.NDim())
	}
	if s.Numel() != 24 {
		t.Errorf("expected 24 elements, got %d", s.Numel())
	}
	if s.At(0) != 2 || s.At(1) != 3 || s.At(2) != 4 {
		t.Errorf("unexpected dims: %v", s.Dims())
	}
}

func TestShapeStrides(t *testing.T) {
	s := NewShape(2, 3, 4)
	strides := s.Strides()
	if len(strides) != 3 {
		t.Fatalf("expected 3 strides, got %d", len(strides))
	}
	// Row-major: [12, 4, 1]
	if strides[0] != 12 || strides[1] != 4 || strides[2] != 1 {
		t.Errorf("unexpected strides: %v", strides)
	}
}

func TestTensorZeros(t *testing.T) {
	tensor := Zeros(NewShape(2, 3), F32)
	if tensor.Shape().Numel() != 6 {
		t.Errorf("expected 6 elements, got %d", tensor.Shape().Numel())
	}
	for _, v := range tensor.Data() {
		if v != 0 {
			t.Errorf("expected 0, got %f", v)
		}
	}
}

func TestTensorOnes(t *testing.T) {
	tensor := Ones(NewShape(2, 3), F32)
	for _, v := range tensor.Data() {
		if v != 1 {
			t.Errorf("expected 1, got %f", v)
		}
	}
}

func TestTensorFromSlice(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor := FromSlice(data, NewShape(2, 3))
	if tensor.At(0, 0) != 1 || tensor.At(1, 2) != 6 {
		t.Errorf("unexpected values")
	}
}

func TestTensorAdd(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	b := FromSlice([]float32{4, 5, 6}, NewShape(3))
	c := a.Add(b)
	data := c.Data()
	if data[0] != 5 || data[1] != 7 || data[2] != 9 {
		t.Errorf("unexpected sum: %v", data)
	}
}

func TestTensorMul(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	b := FromSlice([]float32{4, 5, 6}, NewShape(3))
	c := a.Mul(b)
	data := c.Data()
	if data[0] != 4 || data[1] != 10 || data[2] != 18 {
		t.Errorf("unexpected product: %v", data)
	}
}

func TestTensorScale(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(3))
	c := a.Scale(2)
	data := c.Data()
	if data[0] != 2 || data[1] != 4 || data[2] != 6 {
		t.Errorf("unexpected scaled: %v", data)
	}
}

func TestTensorSiLU(t *testing.T) {
	a := FromSlice([]float32{0, 1, -1}, NewShape(3))
	c := a.SiLU()
	data := c.Data()
	// SiLU(0) = 0, SiLU(1) ≈ 0.731, SiLU(-1) ≈ -0.269
	if math.Abs(float64(data[0])) > 0.001 {
		t.Errorf("expected ~0, got %f", data[0])
	}
	if math.Abs(float64(data[1])-0.731) > 0.01 {
		t.Errorf("expected ~0.731, got %f", data[1])
	}
}

func TestTensorSoftmax(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3}, NewShape(1, 3))
	c := a.Softmax()
	data := c.Data()
	sum := data[0] + data[1] + data[2]
	if math.Abs(float64(sum)-1.0) > 0.001 {
		t.Errorf("expected sum 1, got %f", sum)
	}
	// Should be monotonically increasing
	if data[0] >= data[1] || data[1] >= data[2] {
		t.Errorf("expected monotonic increase: %v", data)
	}
}

func TestMatmul(t *testing.T) {
	// [2, 3] x [3, 4] -> [2, 4]
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, NewShape(2, 3))
	b := FromSlice([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, NewShape(3, 4))
	c := Matmul(a, b)

	if !c.Shape().Equal(NewShape(2, 4)) {
		t.Errorf("unexpected shape: %v", c.Shape())
	}

	// c[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
	if c.At(0, 0) != 38 {
		t.Errorf("expected 38, got %f", c.At(0, 0))
	}
}

func TestTranspose(t *testing.T) {
	a := FromSlice([]float32{1, 2, 3, 4, 5, 6}, NewShape(2, 3))
	b := a.Transpose()
	if !b.Shape().Equal(NewShape(3, 2)) {
		t.Errorf("unexpected shape: %v", b.Shape())
	}
	if b.At(0, 0) != 1 || b.At(0, 1) != 4 || b.At(1, 0) != 2 {
		t.Errorf("unexpected values after transpose")
	}
}

func TestDType(t *testing.T) {
	if F32.Size() != 4 {
		t.Errorf("expected F32 size 4, got %d", F32.Size())
	}
	if F16.Size() != 2 {
		t.Errorf("expected F16 size 2, got %d", F16.Size())
	}
	if F32.String() != "f32" {
		t.Errorf("expected 'f32', got '%s'", F32.String())
	}
}

func TestBroadcast(t *testing.T) {
	a := NewShape(3, 1, 5)
	b := NewShape(4, 5)
	c, err := Broadcast(a, b)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !c.Equal(NewShape(3, 4, 5)) {
		t.Errorf("expected [3,4,5], got %v", c)
	}
}

func TestBroadcastError(t *testing.T) {
	a := NewShape(3, 4)
	b := NewShape(5, 4)
	_, err := Broadcast(a, b)
	if err == nil {
		t.Error("expected broadcast error")
	}
}
