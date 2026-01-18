// Package cuda tests for CUDA FFI bindings.
//
// These tests verify that:
// 1. The stub library returns ErrCudaNotAvailable when CUDA is not present
// 2. Input validation works correctly
// 3. All exported functions have correct signatures
package cuda

import (
	"errors"
	"testing"
)

// =============================================================================
// Basic type tests
// =============================================================================

func TestDefaultStream(t *testing.T) {
	if DefaultStream != nil {
		t.Error("DefaultStream should be nil")
	}
}

func TestErrCudaNotAvailable(t *testing.T) {
	if ErrCudaNotAvailable == nil {
		t.Error("ErrCudaNotAvailable should not be nil")
	}
	if ErrCudaNotAvailable.Error() != "CUDA not available" {
		t.Errorf("unexpected error message: %s", ErrCudaNotAvailable.Error())
	}
}

// =============================================================================
// Input validation tests
// =============================================================================

func TestSiLUInputValidation(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	output := []float32{0.0, 0.0} // wrong length

	err := SiLU(input, output, DefaultStream)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
	if err.Error() != "input and output must have same length" {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAddInputValidation(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{1.0, 2.0}
	output := []float32{0.0, 0.0, 0.0}

	err := Add(a, b, output, DefaultStream)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
}

func TestMulInputValidation(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{1.0, 2.0, 3.0}
	output := []float32{0.0, 0.0} // wrong length

	err := Mul(a, b, output, DefaultStream)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
}

func TestScaleInputValidation(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0}
	output := []float32{0.0} // wrong length

	err := Scale(input, output, 2.0, DefaultStream)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
}

// =============================================================================
// Stub tests (CUDA not available)
// These tests verify that all FFI functions properly return ErrCudaNotAvailable
// when using the stub library (no CUDA).
// =============================================================================

func TestSiLUStub(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	output := make([]float32, 4)

	err := SiLU(input, output, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestAddStub(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{5.0, 6.0, 7.0, 8.0}
	output := make([]float32, 4)

	err := Add(a, b, output, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestMulStub(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{5.0, 6.0, 7.0, 8.0}
	output := make([]float32, 4)

	err := Mul(a, b, output, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestScaleStub(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	output := make([]float32, 4)

	err := Scale(input, output, 2.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestSoftmaxStub(t *testing.T) {
	batch, dim := 2, 4
	input := make([]float32, batch*dim)
	output := make([]float32, batch*dim)

	err := Softmax(input, output, batch, dim, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestRMSNormStub(t *testing.T) {
	batch, dim := 2, 4
	input := make([]float32, batch*dim)
	weight := make([]float32, dim)
	output := make([]float32, batch*dim)

	err := RMSNorm(input, weight, output, batch, dim, 1e-5, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestGEMMStub(t *testing.T) {
	M, N, K := 2, 3, 4
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	C := make([]float32, M*N)

	err := GEMM(A, B, C, M, N, K, 1.0, 0.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestGEMMBatchedStub(t *testing.T) {
	batch, M, N, K := 2, 2, 3, 4
	A := make([]float32, batch*M*K)
	B := make([]float32, batch*K*N)
	C := make([]float32, batch*M*N)

	err := GEMMBatched(A, B, C, batch, M, N, K, 1.0, 0.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestCrossEntropyForwardStub(t *testing.T) {
	batch, vocabSize := 2, 10
	logits := make([]float32, batch*vocabSize)
	targets := make([]int32, batch)
	loss := make([]float32, batch)
	logProbs := make([]float32, batch*vocabSize)

	err := CrossEntropyForward(logits, targets, loss, logProbs, batch, vocabSize, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestAdamWStepStub(t *testing.T) {
	size := 10
	param := make([]float32, size)
	grad := make([]float32, size)
	m := make([]float32, size)
	v := make([]float32, size)

	err := AdamWStep(param, grad, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestArgmaxStub(t *testing.T) {
	batch, vocabSize := 2, 10
	logits := make([]float32, batch*vocabSize)
	output := make([]int32, batch)

	err := Argmax(logits, output, batch, vocabSize, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestSampleStub(t *testing.T) {
	batch, vocabSize := 2, 10
	logits := make([]float32, batch*vocabSize)
	output := make([]int32, batch)
	seeds := make([]uint64, batch)

	err := Sample(logits, output, seeds, batch, vocabSize, 1.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestTopKSampleStub(t *testing.T) {
	batch, vocabSize, k := 2, 10, 5
	logits := make([]float32, batch*vocabSize)
	output := make([]int32, batch)
	seeds := make([]uint64, batch)

	err := TopKSample(logits, output, seeds, batch, vocabSize, k, 1.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

func TestTopPSampleStub(t *testing.T) {
	batch, vocabSize := 2, 10
	logits := make([]float32, batch*vocabSize)
	output := make([]int32, batch)
	seeds := make([]uint64, batch)

	err := TopPSample(logits, output, seeds, batch, vocabSize, 0.9, 1.0, DefaultStream)
	if !errors.Is(err, ErrCudaNotAvailable) {
		t.Errorf("expected ErrCudaNotAvailable, got: %v", err)
	}
}

// =============================================================================
// Benchmark stubs (for future use when CUDA is available)
// =============================================================================

func BenchmarkSiLU(b *testing.B) {
	input := make([]float32, 1024)
	output := make([]float32, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(input, output, DefaultStream)
	}
}

func BenchmarkGEMM(b *testing.B) {
	M, N, K := 128, 128, 128
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	C := make([]float32, M*N)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = GEMM(A, B, C, M, N, K, 1.0, 0.0, DefaultStream)
	}
}
