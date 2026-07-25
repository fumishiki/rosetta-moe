// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Copyright (c) 2025-2026 fumi-engineer

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include "metal_bridge.h"

// Device management
void* metal_create_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return NULL;
    return (__bridge_retained void*)device;
}

void* metal_create_command_queue(void* device_ptr) {
    if (!device_ptr) return NULL;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return NULL;
    return (__bridge_retained void*)queue;
}

// Buffer management (unified memory = zero-copy)
void* metal_create_buffer(void* device_ptr, const void* data, size_t length) {
    if (!device_ptr || !data || length == 0) return NULL;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLBuffer> buffer = [device newBufferWithBytes:data
                                               length:length
                                              options:MTLResourceStorageModeShared];
    if (!buffer) return NULL;
    return (__bridge_retained void*)buffer;
}

void* metal_buffer_contents(void* buffer_ptr) {
    if (!buffer_ptr) return NULL;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
    return [buffer contents];
}

// MPS MatrixMultiplication
// Matrix layout: row-major, C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
void metal_mps_matmul(void* device_ptr, void* queue_ptr,
                       void* a_buf, void* b_buf, void* c_buf,
                       int M, int N, int K) {
    if (!device_ptr || !queue_ptr || !a_buf || !b_buf || !c_buf) return;
    if (M <= 0 || N <= 0 || K <= 0) return;

    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)a_buf;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)b_buf;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)c_buf;

    // MPS uses column-major by default, but we can control via rowBytes
    // Row-major: rowBytes = cols * sizeof(float)
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
        columns:K
        rowBytes:K * sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:K
        columns:N
        rowBytes:N * sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
        columns:N
        rowBytes:N * sizeof(float)
        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:descC];

    // C = alpha * A @ B + beta * C
    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
                                       initWithDevice:device
                                       transposeLeft:NO
                                       transposeRight:NO
                                       resultRows:M
                                       resultColumns:N
                                       interiorColumns:K
                                       alpha:1.0
                                       beta:0.0];

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    [matmul encodeToCommandBuffer:cmdBuf
                      leftMatrix:matA
                     rightMatrix:matB
                    resultMatrix:matC];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

// Custom kernel dispatch from MSL file
void* metal_load_library(void* device_ptr, const char* path) {
    if (!device_ptr || !path) return NULL;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    NSString *nsPath = [NSString stringWithUTF8String:path];
    NSError *error = nil;
    NSURL *url = [NSURL fileURLWithPath:nsPath];
    id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
    if (error || !lib) return NULL;
    return (__bridge_retained void*)lib;
}

void* metal_load_library_from_source(void* device_ptr, const char* source) {
    if (!device_ptr || !source) return NULL;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    NSString *nsSource = [NSString stringWithUTF8String:source];
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsSource options:nil error:&error];
    if (error || !lib) return NULL;
    return (__bridge_retained void*)lib;
}

void* metal_create_function(void* library_ptr, const char* fn_name) {
    if (!library_ptr || !fn_name) return NULL;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library_ptr;
    NSString *nsFnName = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [lib newFunctionWithName:nsFnName];
    if (!fn) return NULL;
    return (__bridge_retained void*)fn;
}

void* metal_create_pipeline(void* device_ptr, void* function_ptr) {
    if (!device_ptr || !function_ptr) return NULL;
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLFunction> fn = (__bridge id<MTLFunction>)function_ptr;
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    if (error || !pipeline) return NULL;
    return (__bridge_retained void*)pipeline;
}

void metal_dispatch_kernel(void* queue_ptr, void* pipeline_ptr,
                            void** buffers, int n_buffers,
                            int grid_x, int grid_y, int grid_z,
                            int threadgroup_x, int threadgroup_y, int threadgroup_z) {
    if (!queue_ptr || !pipeline_ptr || !buffers || n_buffers <= 0) return;
    if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0) return;
    if (threadgroup_x <= 0 || threadgroup_y <= 0 || threadgroup_z <= 0) return;

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pipeline_ptr;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Bind buffers
    for (int i = 0; i < n_buffers; i++) {
        if (buffers[i]) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[i];
            [encoder setBuffer:buf offset:0 atIndex:i];
        }
    }

    MTLSize gridSize = MTLSizeMake(grid_x, grid_y, grid_z);
    MTLSize threadgroupSize = MTLSizeMake(threadgroup_x, threadgroup_y, threadgroup_z);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

// Memory management
void metal_release(void* ptr) {
    if (ptr) {
        CFRelease(ptr);
    }
}
