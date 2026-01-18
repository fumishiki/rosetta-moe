// Package main provides benchmark utilities for cross-language comparison
package main

import "math"

// Matmul performs naive matrix multiplication: C = A @ B
func Matmul(a, b []float32, m, k, n int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// Softmax performs row-wise softmax
func Softmax(input []float32, rows, cols int) []float32 {
	output := make([]float32, len(input))
	for r := 0; r < rows; r++ {
		offset := r * cols

		// Find max for numerical stability
		maxVal := input[offset]
		for c := 1; c < cols; c++ {
			if input[offset+c] > maxVal {
				maxVal = input[offset+c]
			}
		}

		// Compute exp and sum
		var sum float32
		for c := 0; c < cols; c++ {
			expVal := float32(math.Exp(float64(input[offset+c] - maxVal)))
			output[offset+c] = expVal
			sum += expVal
		}

		// Normalize
		for c := 0; c < cols; c++ {
			output[offset+c] /= sum
		}
	}
	return output
}

// SiLU applies SiLU activation: x * sigmoid(x)
func SiLU(input []float32) []float32 {
	output := make([]float32, len(input))
	for i, x := range input {
		output[i] = x * (1.0 / (1.0 + float32(math.Exp(float64(-x)))))
	}
	return output
}

// RMSNorm applies RMS normalization
func RMSNorm(input, weight []float32, dim int, eps float32) []float32 {
	n := len(input) / dim
	output := make([]float32, len(input))

	for i := 0; i < n; i++ {
		offset := i * dim

		// Compute RMS
		var sumSq float32
		for j := 0; j < dim; j++ {
			x := input[offset+j]
			sumSq += x * x
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(dim) + eps)))

		// Normalize and scale
		for j := 0; j < dim; j++ {
			output[offset+j] = (input[offset+j] / rms) * weight[j]
		}
	}
	return output
}

// RandomVec generates a random float32 slice using LCG
func RandomVec(size int, seed uint64) []float32 {
	state := seed
	result := make([]float32, size)
	for i := 0; i < size; i++ {
		state = state*6364136223846793005 + 1
		result[i] = float32(state>>33)/float32(^uint32(0))*2.0 - 1.0
	}
	return result
}

func main() {}
