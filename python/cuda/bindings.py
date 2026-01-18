"""ctypes bindings to CUDA kernels."""

from __future__ import annotations

import ctypes
from ctypes import c_float, c_int, c_int32, c_int64, c_uint64, POINTER
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CudaError(Exception):
    """CUDA operation error."""

    pass


# Try to load the CUDA library
_lib: ctypes.CDLL | None = None
_cuda_available = False

# Look for library in standard locations
_lib_paths = [
    Path(__file__).parent.parent.parent / "go" / "cuda" / "lib" / "libcudann.a",
    Path(__file__).parent.parent.parent / "go" / "cuda" / "lib" / "libcudann.so",
    Path(__file__).parent.parent.parent / "go" / "cuda" / "lib" / "libcudann.dylib",
]


def _load_library() -> ctypes.CDLL | None:
    """Try to load the CUDA library."""
    global _lib, _cuda_available

    for lib_path in _lib_paths:
        if lib_path.exists():
            try:
                # Note: .a files need different handling
                if lib_path.suffix == ".a":
                    # Static libraries can't be loaded directly with ctypes
                    # Would need a shared library wrapper
                    continue
                _lib = ctypes.CDLL(str(lib_path))
                _cuda_available = True
                return _lib
            except OSError:
                continue

    return None


# Try to load on import
_load_library()


def cuda_available() -> bool:
    """Check if CUDA is available."""
    return _cuda_available


def _check_result(result: int) -> None:
    """Check CUDA function return code."""
    if result != 0:
        raise CudaError("CUDA operation failed")


def _as_float_ptr(arr: NDArray[np.float32]) -> POINTER(c_float):
    """Convert numpy array to float pointer."""
    return arr.ctypes.data_as(POINTER(c_float))


def _as_int32_ptr(arr: NDArray[np.int32]) -> POINTER(c_int32):
    """Convert numpy array to int32 pointer."""
    return arr.ctypes.data_as(POINTER(c_int32))


def _as_uint64_ptr(arr: NDArray[np.uint64]) -> POINTER(c_uint64):
    """Convert numpy array to uint64 pointer."""
    return arr.ctypes.data_as(POINTER(c_uint64))


# CPU fallback implementations


def silu(input_arr: NDArray[np.float32], output: NDArray[np.float32]) -> None:
    """SiLU activation: x * sigmoid(x)."""
    if _lib is not None:
        _check_result(
            _lib.cuda_silu(
                _as_float_ptr(input_arr),
                _as_float_ptr(output),
                c_int64(input_arr.size),
                None,
            )
        )
    else:
        # CPU fallback
        sigmoid = 1.0 / (1.0 + np.exp(-input_arr))
        np.copyto(output, input_arr * sigmoid)


def add(
    a: NDArray[np.float32], b: NDArray[np.float32], output: NDArray[np.float32]
) -> None:
    """Element-wise addition."""
    if _lib is not None:
        _check_result(
            _lib.cuda_add(
                _as_float_ptr(a),
                _as_float_ptr(b),
                _as_float_ptr(output),
                c_int64(a.size),
                None,
            )
        )
    else:
        np.copyto(output, a + b)


def mul(
    a: NDArray[np.float32], b: NDArray[np.float32], output: NDArray[np.float32]
) -> None:
    """Element-wise multiplication."""
    if _lib is not None:
        _check_result(
            _lib.cuda_mul(
                _as_float_ptr(a),
                _as_float_ptr(b),
                _as_float_ptr(output),
                c_int64(a.size),
                None,
            )
        )
    else:
        np.copyto(output, a * b)


def scale(
    input_arr: NDArray[np.float32], output: NDArray[np.float32], scalar: float
) -> None:
    """Scale by scalar."""
    if _lib is not None:
        _check_result(
            _lib.cuda_scale(
                _as_float_ptr(input_arr),
                _as_float_ptr(output),
                c_float(scalar),
                c_int64(input_arr.size),
                None,
            )
        )
    else:
        np.copyto(output, input_arr * scalar)


def softmax(
    input_arr: NDArray[np.float32],
    output: NDArray[np.float32],
    batch: int,
    dim: int,
) -> None:
    """Softmax."""
    if _lib is not None:
        _check_result(
            _lib.cuda_softmax(
                _as_float_ptr(input_arr),
                _as_float_ptr(output),
                c_int(batch),
                c_int(dim),
                None,
            )
        )
    else:
        # CPU fallback
        reshaped = input_arr.reshape(batch, dim)
        max_vals = np.max(reshaped, axis=-1, keepdims=True)
        exp_vals = np.exp(reshaped - max_vals)
        result = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        np.copyto(output, result.ravel())


def rmsnorm(
    input_arr: NDArray[np.float32],
    weight: NDArray[np.float32],
    output: NDArray[np.float32],
    batch: int,
    dim: int,
    eps: float = 1e-6,
) -> None:
    """RMS normalization."""
    if _lib is not None:
        _check_result(
            _lib.cuda_rmsnorm(
                _as_float_ptr(input_arr),
                _as_float_ptr(weight),
                _as_float_ptr(output),
                c_int(batch),
                c_int(dim),
                c_float(eps),
                None,
            )
        )
    else:
        # CPU fallback
        reshaped = input_arr.reshape(batch, dim)
        rms = np.sqrt(np.mean(reshaped**2, axis=-1, keepdims=True) + eps)
        normalized = reshaped / rms
        result = normalized * weight
        np.copyto(output, result.ravel())


def gemm(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    c: NDArray[np.float32],
    m: int,
    n: int,
    k: int,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    """Matrix multiplication: C = alpha * A @ B + beta * C."""
    if _lib is not None:
        _check_result(
            _lib.cuda_gemm(
                _as_float_ptr(a),
                _as_float_ptr(b),
                _as_float_ptr(c),
                c_int(m),
                c_int(n),
                c_int(k),
                c_float(alpha),
                c_float(beta),
                None,
            )
        )
    else:
        # CPU fallback
        a_mat = a.reshape(m, k)
        b_mat = b.reshape(k, n)
        c_mat = c.reshape(m, n)
        result = alpha * (a_mat @ b_mat) + beta * c_mat
        np.copyto(c, result.ravel())


def cross_entropy_forward(
    logits: NDArray[np.float32],
    targets: NDArray[np.int32],
    loss: NDArray[np.float32],
    log_probs: NDArray[np.float32],
    batch: int,
    vocab_size: int,
) -> None:
    """Cross entropy loss forward."""
    if _lib is not None:
        _check_result(
            _lib.cuda_cross_entropy_forward(
                _as_float_ptr(logits),
                _as_int32_ptr(targets),
                _as_float_ptr(loss),
                _as_float_ptr(log_probs),
                c_int(batch),
                c_int(vocab_size),
                None,
            )
        )
    else:
        # CPU fallback
        logits_2d = logits.reshape(batch, vocab_size)
        max_logits = np.max(logits_2d, axis=-1, keepdims=True)
        shifted = logits_2d - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs_2d = shifted - log_sum_exp
        np.copyto(log_probs, log_probs_2d.ravel())
        loss[0] = -np.mean(log_probs_2d[np.arange(batch), targets])


def adamw_step(
    param: NDArray[np.float32],
    grad: NDArray[np.float32],
    m: NDArray[np.float32],
    v: NDArray[np.float32],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> None:
    """AdamW optimizer step."""
    if _lib is not None:
        _check_result(
            _lib.cuda_adamw_step(
                _as_float_ptr(param),
                _as_float_ptr(grad),
                _as_float_ptr(m),
                _as_float_ptr(v),
                c_float(lr),
                c_float(beta1),
                c_float(beta2),
                c_float(eps),
                c_float(weight_decay),
                c_int(step),
                c_int64(param.size),
                None,
            )
        )
    else:
        # CPU fallback
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        param[:] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * param)


def argmax(
    logits: NDArray[np.float32],
    output: NDArray[np.int32],
    batch: int,
    vocab_size: int,
) -> None:
    """Argmax for greedy decoding."""
    if _lib is not None:
        _check_result(
            _lib.cuda_argmax(
                _as_float_ptr(logits),
                _as_int32_ptr(output),
                c_int(batch),
                c_int(vocab_size),
                None,
            )
        )
    else:
        # CPU fallback
        logits_2d = logits.reshape(batch, vocab_size)
        output[:] = np.argmax(logits_2d, axis=-1).astype(np.int32)


def sample(
    logits: NDArray[np.float32],
    output: NDArray[np.int32],
    seeds: NDArray[np.uint64],
    batch: int,
    vocab_size: int,
    temperature: float = 1.0,
) -> None:
    """Multinomial sampling."""
    if _lib is not None:
        _check_result(
            _lib.cuda_sample(
                _as_float_ptr(logits),
                _as_int32_ptr(output),
                _as_uint64_ptr(seeds),
                c_int(batch),
                c_int(vocab_size),
                c_float(temperature),
                None,
            )
        )
    else:
        # CPU fallback
        logits_2d = logits.reshape(batch, vocab_size) / temperature
        max_logits = np.max(logits_2d, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_2d - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        for i in range(batch):
            output[i] = np.random.choice(vocab_size, p=probs[i])


def topk_sample(
    logits: NDArray[np.float32],
    output: NDArray[np.int32],
    seeds: NDArray[np.uint64],
    batch: int,
    vocab_size: int,
    k: int,
    temperature: float = 1.0,
) -> None:
    """Top-k sampling."""
    if _lib is not None:
        _check_result(
            _lib.cuda_topk_sample(
                _as_float_ptr(logits),
                _as_int32_ptr(output),
                _as_uint64_ptr(seeds),
                c_int(batch),
                c_int(vocab_size),
                c_int(k),
                c_float(temperature),
                None,
            )
        )
    else:
        # CPU fallback
        logits_2d = logits.reshape(batch, vocab_size) / temperature
        for i in range(batch):
            top_k_idx = np.argsort(logits_2d[i])[-k:]
            top_k_logits = logits_2d[i, top_k_idx]
            exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
            probs = exp_logits / np.sum(exp_logits)
            output[i] = top_k_idx[np.random.choice(k, p=probs)]


def topp_sample(
    logits: NDArray[np.float32],
    output: NDArray[np.int32],
    seeds: NDArray[np.uint64],
    batch: int,
    vocab_size: int,
    top_p: float,
    temperature: float = 1.0,
) -> None:
    """Nucleus (top-p) sampling."""
    if _lib is not None:
        _check_result(
            _lib.cuda_topp_sample(
                _as_float_ptr(logits),
                _as_int32_ptr(output),
                _as_uint64_ptr(seeds),
                c_int(batch),
                c_int(vocab_size),
                c_float(top_p),
                c_float(temperature),
                None,
            )
        )
    else:
        # CPU fallback
        logits_2d = logits.reshape(batch, vocab_size) / temperature
        for i in range(batch):
            sorted_idx = np.argsort(logits_2d[i])[::-1]
            sorted_logits = logits_2d[i, sorted_idx]
            exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
            probs = exp_logits / np.sum(exp_logits)
            cumsum = np.cumsum(probs)
            cutoff = np.searchsorted(cumsum, top_p) + 1
            top_p_idx = sorted_idx[:cutoff]
            top_p_probs = probs[:cutoff]
            top_p_probs /= np.sum(top_p_probs)
            output[i] = top_p_idx[np.random.choice(cutoff, p=top_p_probs)]
