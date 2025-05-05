
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"
__device__ int device_strcmp22(const char *s1, const char *s2);

template <typename T>
__global__ void hashJoinKernel(
    const T *__restrict__ left_data,
    const T *__restrict__ right_data,
    size_t left_size,
    size_t right_size,
    size_t *__restrict__ out_left_idx,
    size_t *__restrict__ out_right_idx,
    unsigned long long *__restrict__ out_count);

template <typename T>
__global__ void getRowsKernel(
    const T *__restrict__ in,
    const size_t *__restrict__ idx,
    T *__restrict__ out,
    size_t out_size);