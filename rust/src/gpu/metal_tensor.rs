// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

//! Metal GPU context and tensor types.
//!
//! MetalContext: device + command queue + compiled shader library.
//! MetalTensor: MTLBuffer-backed GPU tensor with unified memory.
//!
//! All GPU data lives in MetalTensor (MTLBuffer). No `Vec<f32>` storage.

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

/// Host-side buffer returned by GPU readback operations.
pub type HostBuffer<T = f32> = Vec<T>;

// FFI declarations matching metal_bridge.h
unsafe extern "C" {
    fn metal_create_device() -> *mut c_void;
    fn metal_create_command_queue(device: *mut c_void) -> *mut c_void;
    fn metal_release(ptr: *mut c_void);
    fn metal_create_buffer(device: *mut c_void, data: *const c_void, length: usize) -> *mut c_void;
    fn metal_buffer_contents(buffer: *mut c_void) -> *mut c_void;
    pub(super) fn metal_mps_matmul(
        device: *mut c_void,
        queue: *mut c_void,
        a_buf: *mut c_void,
        b_buf: *mut c_void,
        c_buf: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
    );
    fn metal_load_library(device: *mut c_void, path: *const c_char) -> *mut c_void;
    fn metal_load_library_from_source(device: *mut c_void, source: *const c_char) -> *mut c_void;
    fn metal_create_function(library: *mut c_void, fn_name: *const c_char) -> *mut c_void;
    fn metal_create_pipeline(device: *mut c_void, function: *mut c_void) -> *mut c_void;
    fn metal_dispatch_kernel(
        queue: *mut c_void,
        pipeline: *mut c_void,
        buffers: *const *mut c_void,
        n_buffers: c_int,
        grid_x: c_int,
        grid_y: c_int,
        grid_z: c_int,
        threadgroup_x: c_int,
        threadgroup_y: c_int,
        threadgroup_z: c_int,
    );
}

/// Metal GPU context: device, command queue, and compiled compute pipelines.
pub struct MetalContext {
    pub(super) device: *mut c_void,
    pub(super) queue: *mut c_void,
    pub(super) shader_lib: *mut c_void,
}

/// GPU tensor backed by MTLBuffer with unified memory.
pub struct MetalTensor {
    pub(super) buffer: *mut c_void,
    shape: Vec<usize>,
    len: usize,
}

impl MetalContext {
    /// Create Metal context with default device.
    pub fn new() -> Result<Self, String> {
        // SAFETY: FFI call to Objective-C Metal framework.
        // metal_create_device returns retained MTLDevice or NULL.
        let device = unsafe { metal_create_device() };
        if device.is_null() {
            return Err("No Metal device found".to_string());
        }

        // SAFETY: device is a valid MTLDevice pointer.
        let queue = unsafe { metal_create_command_queue(device) };
        if queue.is_null() {
            // SAFETY: device was retained, release it on failure.
            unsafe { metal_release(device) };
            return Err("Failed to create Metal command queue".to_string());
        }

        Ok(Self {
            device,
            queue,
            shader_lib: ptr::null_mut(),
        })
    }

    /// Load a compiled .metallib shader library.
    pub fn load_shader_library(&mut self, path: &str) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;

        // SAFETY: device is valid, c_path is null-terminated.
        let lib = unsafe { metal_load_library(self.device, c_path.as_ptr()) };
        if lib.is_null() {
            return Err(format!("Failed to load Metal library: {path}"));
        }

        // Release previous library if any.
        if !self.shader_lib.is_null() {
            // SAFETY: shader_lib was retained by metal_load_library.
            unsafe { metal_release(self.shader_lib) };
        }
        self.shader_lib = lib;
        Ok(())
    }

    /// Compile and load Metal shader library from source code.
    pub fn load_shader_source(&mut self, source: &str) -> Result<(), String> {
        let c_source = CString::new(source).map_err(|e| e.to_string())?;

        // SAFETY: device is valid, c_source is null-terminated MSL source.
        let lib = unsafe { metal_load_library_from_source(self.device, c_source.as_ptr()) };
        if lib.is_null() {
            return Err("Failed to compile Metal shader from source".to_string());
        }

        // Release previous library if any.
        if !self.shader_lib.is_null() {
            // SAFETY: shader_lib was retained by metal_load_library_from_source.
            unsafe { metal_release(self.shader_lib) };
        }
        self.shader_lib = lib;
        Ok(())
    }

    /// Load all required shaders for MoE forward pass from source files.
    pub fn load_required_shaders(&mut self) -> Result<(), String> {
        // Combine all required shader source files
        let rmsnorm_src = include_str!("../../../shaders/rmsnorm.metal");
        let silu_src = include_str!("../../../shaders/silu.metal");
        let elementwise_src = include_str!("../../../shaders/elementwise.metal");
        let softmax_src = include_str!("../../../shaders/softmax.metal");
        let attention_src = include_str!("../../../shaders/attention.metal");
        let embedding_src = include_str!("../../../shaders/embedding.metal");

        // Concatenate all sources (Metal allows multiple kernels in one library)
        let combined = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            rmsnorm_src, silu_src, elementwise_src, softmax_src, attention_src, embedding_src
        );
        self.load_shader_source(&combined)
    }

    /// Get raw device pointer (for MetalTensor creation).
    pub(super) fn device_ptr(&self) -> *mut c_void {
        self.device
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // SAFETY: All pointers were retained by their respective create functions.
        // Release in reverse order of creation.
        unsafe {
            if !self.shader_lib.is_null() {
                metal_release(self.shader_lib);
            }
            if !self.queue.is_null() {
                metal_release(self.queue);
            }
            if !self.device.is_null() {
                metal_release(self.device);
            }
        }
    }
}

impl MetalTensor {
    /// Create MetalTensor from f32 slice (copies data to GPU buffer).
    pub fn upload(ctx: &MetalContext, data: &[f32], shape: Vec<usize>) -> Result<Self, String> {
        let byte_len = std::mem::size_of_val(data);

        // SAFETY: data pointer is valid for byte_len bytes, device is valid.
        let buffer = unsafe {
            metal_create_buffer(ctx.device_ptr(), data.as_ptr() as *const c_void, byte_len)
        };
        if buffer.is_null() {
            return Err("Failed to create Metal buffer".to_string());
        }

        Ok(Self {
            buffer,
            shape,
            len: data.len(),
        })
    }

    /// Create zero-initialized MetalTensor.
    pub fn zeros(ctx: &MetalContext, shape: Vec<usize>) -> Result<Self, String> {
        let len: usize = shape.iter().product();
        let data = vec![0.0f32; len];
        Self::upload(ctx, &data, shape)
    }

    /// Create MetalTensor from u32 scalar (for passing params to MSL kernels).
    pub fn from_u32_scalar(ctx: &MetalContext, val: u32) -> Result<Self, String> {
        let bytes = val.to_ne_bytes();
        // SAFETY: 4 bytes for a uint32
        let buffer = unsafe {
            metal_create_buffer(ctx.device_ptr(), bytes.as_ptr() as *const c_void, 4)
        };
        if buffer.is_null() {
            return Err("Failed to create u32 buffer".to_string());
        }
        Ok(Self {
            buffer,
            shape: vec![1],
            len: 1,
        })
    }

    /// Create MetalTensor from f32 scalar (for passing params to MSL kernels).
    pub fn from_f32_scalar(ctx: &MetalContext, val: f32) -> Result<Self, String> {
        let bytes = val.to_ne_bytes();
        // SAFETY: 4 bytes for a float32
        let buffer = unsafe {
            metal_create_buffer(ctx.device_ptr(), bytes.as_ptr() as *const c_void, 4)
        };
        if buffer.is_null() {
            return Err("Failed to create f32 buffer".to_string());
        }
        Ok(Self {
            buffer,
            shape: vec![1],
            len: 1,
        })
    }

    /// Read GPU buffer back to host memory.
    pub fn download(&self) -> HostBuffer {
        if self.buffer.is_null() || self.len == 0 {
            return vec![];
        }

        // SAFETY: buffer is a valid MTLBuffer with StorageModeShared.
        // contents() returns a raw pointer to unified memory.
        let ptr = unsafe { metal_buffer_contents(self.buffer) };
        if ptr.is_null() {
            return vec![];
        }

        let mut out = vec![0.0f32; self.len];
        // SAFETY: ptr points to at least self.len * sizeof(f32) bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), self.len);
        }
        out
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw buffer pointer (for kernel dispatch).
    pub(super) fn buffer_ptr(&self) -> *mut c_void {
        self.buffer
    }

    /// Get pointer to unified memory contents (zero-copy on Apple Silicon).
    /// Returns null if buffer is invalid.
    pub(super) fn contents_ptr(&self) -> *mut f32 {
        if self.buffer.is_null() {
            return ptr::null_mut();
        }
        // SAFETY: buffer is a valid MTLBuffer with StorageModeShared.
        unsafe { metal_buffer_contents(self.buffer) as *mut f32 }
    }

    /// Read a single f32 from GPU buffer at the given index.
    pub fn read_f32(&self, index: usize) -> f32 {
        let ptr = self.contents_ptr();
        if ptr.is_null() || index >= self.len {
            return 0.0;
        }
        // SAFETY: ptr is valid unified memory, index is in bounds.
        unsafe { ptr.add(index).read() }
    }
}

impl Drop for MetalTensor {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            // SAFETY: buffer was retained by metal_create_buffer.
            unsafe { metal_release(self.buffer) };
        }
    }
}

/// Matrix multiplication using MPS (Metal Performance Shaders).
/// C = A @ B  where A: [M,K], B: [K,N], C: [M,N] (row-major).
pub fn mps_matmul(
    ctx: &MetalContext,
    a: &MetalTensor,
    b: &MetalTensor,
    out: &mut MetalTensor,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), String> {
    if a.buffer.is_null() || b.buffer.is_null() || out.buffer.is_null() {
        return Err("Invalid buffer in mps_matmul".to_string());
    }

    // SAFETY: All pointers are valid, dimensions match buffer sizes.
    unsafe {
        metal_mps_matmul(
            ctx.device,
            ctx.queue,
            a.buffer,
            b.buffer,
            out.buffer,
            m as c_int,
            n as c_int,
            k as c_int,
        );
    }
    Ok(())
}

/// Dispatch a custom Metal compute kernel from the loaded shader library.
pub fn dispatch_kernel(
    ctx: &MetalContext,
    kernel_name: &str,
    buffers: &[&MetalTensor],
    _params: &[u32],
    grid_size: usize,
    threadgroup_size: usize,
) -> Result<(), String> {
    if ctx.shader_lib.is_null() {
        return Err("No shader library loaded".to_string());
    }

    let c_name = CString::new(kernel_name).map_err(|e| e.to_string())?;

    // SAFETY: shader_lib is a valid MTLLibrary.
    let function = unsafe { metal_create_function(ctx.shader_lib, c_name.as_ptr()) };
    if function.is_null() {
        return Err(format!("Failed to create Metal function: {kernel_name}"));
    }

    // SAFETY: device and function are valid.
    let pipeline = unsafe { metal_create_pipeline(ctx.device, function) };
    // SAFETY: function was retained, release it now.
    unsafe { metal_release(function) };

    if pipeline.is_null() {
        return Err(format!("Failed to create pipeline for: {kernel_name}"));
    }

    // Collect raw buffer pointers.
    let buf_ptrs: Vec<*mut c_void> = buffers.iter().map(|t| t.buffer_ptr()).collect();
    let buf_ptr = if buf_ptrs.is_empty() {
        ptr::null()
    } else {
        buf_ptrs.as_ptr()
    };

    // SAFETY: queue, pipeline, and buffers are all valid.
    unsafe {
        metal_dispatch_kernel(
            ctx.queue,
            pipeline,
            buf_ptr,
            buf_ptrs.len() as c_int,
            grid_size as c_int,
            1,
            1,
            threadgroup_size as c_int,
            1,
            1,
        );
        metal_release(pipeline);
    }

    Ok(())
}
