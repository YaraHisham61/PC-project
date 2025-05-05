#pragma once
#include <cuda_runtime.h>

__global__ void findSumElement(float *input, float *output, int size);