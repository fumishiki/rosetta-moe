# Makefile for building CUDA library for Go

CUDA_PATH ?= /usr/local/cuda
CUDA_KERNELS = ../../cuda/kernels
STUB_SRC = ../../cuda/stub.c

# Output
LIB_DIR = lib
LIB_NAME = libcudann.a

# Detect CUDA
ifeq ($(wildcard $(CUDA_PATH)/bin/nvcc),)
    USE_CUDA = 0
else
    USE_CUDA = 1
endif

# Compiler flags
CC = gcc
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -O3 -fPIC -I$(CUDA_KERNELS)
NVCC_FLAGS = -O3 -Xcompiler -fPIC

# GPU architectures
GPU_ARCHS = -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90

# Source files
CUDA_SRCS = $(wildcard $(CUDA_KERNELS)/*.cu)
CUDA_OBJS = $(patsubst $(CUDA_KERNELS)/%.cu,$(LIB_DIR)/%.o,$(CUDA_SRCS))

.PHONY: all clean stub cuda

all: $(LIB_DIR)/$(LIB_NAME)

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

ifeq ($(USE_CUDA),1)
# Build with CUDA
$(LIB_DIR)/$(LIB_NAME): $(CUDA_OBJS) | $(LIB_DIR)
	ar rcs $@ $^
	@echo "Built CUDA library: $@"

$(LIB_DIR)/%.o: $(CUDA_KERNELS)/%.cu | $(LIB_DIR)
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCHS) -c $< -o $@

else
# Build stub (no CUDA)
$(LIB_DIR)/$(LIB_NAME): $(LIB_DIR)/stub.o | $(LIB_DIR)
	ar rcs $@ $^
	@echo "Built stub library (no CUDA): $@"

$(LIB_DIR)/stub.o: $(STUB_SRC) | $(LIB_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

endif

clean:
	rm -rf $(LIB_DIR)

# For debugging
info:
	@echo "USE_CUDA: $(USE_CUDA)"
	@echo "CUDA_PATH: $(CUDA_PATH)"
	@echo "CUDA_SRCS: $(CUDA_SRCS)"
