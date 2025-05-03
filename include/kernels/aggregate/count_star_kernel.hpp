#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>
#include <climits>

template <typename T>
__global__ void countStar(T *input, float *output, int size);