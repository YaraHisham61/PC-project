#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>
#include <climits> // For INT_MIN, FLT_MAX
#include <cstdio>  // Required for printf

template <typename T>
__device__ T atomicMaxGeneric(T *address, T val);

__device__ int cuda_strcmp(const char *s1, const char *s2);
__device__ char *atomicMaxString(char **address, char *new_ptr, const char **strings);

template <typename T>
__global__ void findMaxElement(T *input, T *output, int size);