// Package tensor provides tensor operations for the MoE Transformer.
package tensor

// DType represents the data type of tensor elements.
type DType uint8

const (
	F32 DType = iota
	F16
	BF16
	I32
	I64
)

// Size returns the byte size of each element.
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

// String returns the string representation of the dtype.
func (d DType) String() string {
	switch d {
	case F32:
		return "f32"
	case F16:
		return "f16"
	case BF16:
		return "bf16"
	case I32:
		return "i32"
	case I64:
		return "i64"
	default:
		return "unknown"
	}
}
