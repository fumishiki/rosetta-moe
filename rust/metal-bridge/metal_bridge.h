// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device and command queue management
void* metal_create_device(void);
void* metal_create_command_queue(void* device_ptr);
void metal_release(void* ptr);

// Buffer management (unified memory = zero-copy)
void* metal_create_buffer(void* device_ptr, const void* data, size_t length);
void* metal_buffer_contents(void* buffer_ptr);

// MPS MatrixMultiplication
void metal_mps_matmul(void* device_ptr, void* queue_ptr,
                       void* a_buf, void* b_buf, void* c_buf,
                       int M, int N, int K);

// Custom kernel dispatch from MSL file
void* metal_load_library(void* device_ptr, const char* path);
void* metal_load_library_from_source(void* device_ptr, const char* source);
void* metal_create_function(void* library_ptr, const char* fn_name);
void* metal_create_pipeline(void* device_ptr, void* function_ptr);
void metal_dispatch_kernel(void* queue_ptr, void* pipeline_ptr,
                            void** buffers, int n_buffers,
                            int grid_x, int grid_y, int grid_z,
                            int threadgroup_x, int threadgroup_y, int threadgroup_z);

#ifdef __cplusplus
}
#endif

#endif  // METAL_BRIDGE_H
