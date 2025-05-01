#pragma once
__device__ float atomicMaxFloat(float *address, float val);

__global__ void findMaxElement(float *input, float *output, int size);