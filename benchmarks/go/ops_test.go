package main

import "testing"

// Matmul benchmarks
func BenchmarkMatmul64(b *testing.B) {
	a := RandomVec(64*64, 42)
	bb := RandomVec(64*64, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Matmul(a, bb, 64, 64, 64)
	}
}

func BenchmarkMatmul128(b *testing.B) {
	a := RandomVec(128*128, 42)
	bb := RandomVec(128*128, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Matmul(a, bb, 128, 128, 128)
	}
}

func BenchmarkMatmul256(b *testing.B) {
	a := RandomVec(256*256, 42)
	bb := RandomVec(256*256, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Matmul(a, bb, 256, 256, 256)
	}
}

func BenchmarkMatmul512(b *testing.B) {
	a := RandomVec(512*512, 42)
	bb := RandomVec(512*512, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Matmul(a, bb, 512, 512, 512)
	}
}

// Softmax benchmarks
func BenchmarkSoftmax64x1024(b *testing.B) {
	input := RandomVec(64*1024, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(input, 64, 1024)
	}
}

func BenchmarkSoftmax128x1024(b *testing.B) {
	input := RandomVec(128*1024, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(input, 128, 1024)
	}
}

func BenchmarkSoftmax256x1024(b *testing.B) {
	input := RandomVec(256*1024, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(input, 256, 1024)
	}
}

func BenchmarkSoftmax512x32000(b *testing.B) {
	input := RandomVec(512*32000, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Softmax(input, 512, 32000)
	}
}

// SiLU benchmarks
func BenchmarkSiLU1024(b *testing.B) {
	input := RandomVec(1024, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(input)
	}
}

func BenchmarkSiLU4096(b *testing.B) {
	input := RandomVec(4096, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(input)
	}
}

func BenchmarkSiLU16384(b *testing.B) {
	input := RandomVec(16384, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(input)
	}
}

func BenchmarkSiLU65536(b *testing.B) {
	input := RandomVec(65536, 42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SiLU(input)
	}
}

// RMSNorm benchmarks
func BenchmarkRMSNorm64x768(b *testing.B) {
	input := RandomVec(64*768, 42)
	weight := RandomVec(768, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RMSNorm(input, weight, 768, 1e-6)
	}
}

func BenchmarkRMSNorm128x768(b *testing.B) {
	input := RandomVec(128*768, 42)
	weight := RandomVec(768, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RMSNorm(input, weight, 768, 1e-6)
	}
}

func BenchmarkRMSNorm256x768(b *testing.B) {
	input := RandomVec(256*768, 42)
	weight := RandomVec(768, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RMSNorm(input, weight, 768, 1e-6)
	}
}

func BenchmarkRMSNorm512x768(b *testing.B) {
	input := RandomVec(512*768, 42)
	weight := RandomVec(768, 123)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = RMSNorm(input, weight, 768, 1e-6)
	}
}
