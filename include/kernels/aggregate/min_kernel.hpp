#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>
#include <climits> // For INT_MIN, FLT_MAX
#include <cstdio>  // Required for printf
#include <cfloat>
template <typename T>
__device__ T atomicMinGeneric(T *address, T val);
__device__ int cuda_strcmp_min(const char *s1, const char *s2);
__device__ char *atomicMinString(char **address, char *new_ptr, const char **strings);
template <typename T>
__global__ void findMinElement(T *input, T *output, int size);
