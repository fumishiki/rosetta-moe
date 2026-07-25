// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//go:build darwin

package nn

import "testing"

func TestMetalAvailable(t *testing.T) {
	if !MetalAvailable() {
		t.Skip("Metal not available on this platform")
	}
	t.Log("Metal is available")
}

func TestMetalContext(t *testing.T) {
	ctx := NewMetalContext()
	if ctx == nil {
		t.Skip("Metal not available")
	}
	defer ctx.Close()
	t.Log("Successfully created MetalContext")
}

func TestMetalMatmul(t *testing.T) {
	ctx := NewMetalContext()
	if ctx == nil {
		t.Skip("Metal not available")
	}
	defer ctx.Close()

	// Small matmul test: C = A @ B
	// A: [2, 3], B: [3, 4], C: [2, 4]
	M, N, K := 2, 4, 3

	a := ctx.NewTensor([]float32{
		1, 2, 3,
		4, 5, 6,
	}, M, K)
	defer a.Release()

	b := ctx.NewTensor([]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}, K, N)
	defer b.Release()

	c := ctx.NewTensorZeros(M, N)
	defer c.Release()

	ctx.Matmul(a, b, c, M, N, K)

	result := c.Data()
	if len(result) != M*N {
		t.Fatalf("expected %d elements, got %d", M*N, len(result))
	}

	// Expected result: [[1, 2, 3, 0], [4, 5, 6, 0]]
	expected := []float32{1, 2, 3, 0, 4, 5, 6, 0}
	const eps = 1e-5
	for i, v := range expected {
		if abs32(result[i]-v) > eps {
			t.Errorf("result[%d] = %f, want %f", i, result[i], v)
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
