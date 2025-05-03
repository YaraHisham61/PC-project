
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"
__device__ int device_strcmp22(const char *s1, const char *s2);

template <typename T>
__global__ void hashJoinKernel(
    const T *left_data, const T *right_data,
    bool *output_mask_left, bool *output_mask_right,
    size_t left_size, size_t right_size);