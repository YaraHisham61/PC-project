// filter_kernel.hpp
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"

template <typename T>
__global__ void filterKernel(const T *input, bool *output, size_t row_count, T value, uint8_t cond);

__global__ void filterKernelString(const char **input, bool *output, size_t row_count, const char *value, uint8_t cond);
__device__ int device_strcmp(const char *s1, const char *s2);

__global__ void andKernel(bool *combined_mask, const bool *current_mask, size_t size);
__global__ void orKernel(bool *combined_mask, const bool *current_mask, size_t size);

__global__ void computeOutputPositions(const bool *mask, unsigned int *positions, size_t size);

template <typename T>
__global__ void copySelectedRowsKernel(const T *input, T *output,
                                       const bool *mask, const unsigned int *positions,
                                       size_t size);
__global__ void copySelectedStringRowsKernel(const char **input, const char **output,
                                             const bool *mask, const unsigned int *positions,
                                             size_t size);