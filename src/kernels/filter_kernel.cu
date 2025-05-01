#include "kernels/filter_kernel.hpp"

template <typename T>
__global__ void filterKernel(const T *input, bool *output, size_t row_count, T value, uint8_t cond)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count)
        return;

    switch (cond)
    {
    case 1: // >

        output[idx] = (input[idx] > value) ? true : false;
        break;
    case 2: // <
        output[idx] = (input[idx] < value) ? true : false;
        break;
    case 3: // ==
        output[idx] = (input[idx] == value) ? true : false;
        break;
    case 4: // !=
        output[idx] = (input[idx] != value) ? true : false;
        break;
    case 5: // <=
        output[idx] = (input[idx] <= value) ? true : false;
        break;
    case 6: // >=
        output[idx] = (input[idx] >= value) ? true : false;
        break;
    default:
        output[idx] = false;
        break;
    }
}

__device__ int device_strcmp(const char *s1, const char *s2)
{
    while (*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

__global__ void filterKernelString(const char **input, bool *output, size_t row_count, const char *value, uint8_t cond)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count)
        return;

    output[idx] = (device_strcmp(input[idx], value) == 0);
}

__global__ void andKernel(bool *combined_mask, const bool *current_mask, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        combined_mask[idx] = combined_mask[idx] && current_mask[idx];
    }
}

__global__ void orKernel(bool *combined_mask, const bool *current_mask, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        combined_mask[idx] = combined_mask[idx] || current_mask[idx];
    }
}

template <typename T>
__global__ void filterColumnKernel(const T *input, T *output, const bool *mask,
                                   const size_t row_count, unsigned long long *filtered_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count)
        return;

    if (mask[idx])
    {
        size_t pos = atomicAdd(filtered_count, 1ull);
        if (pos < row_count)
        { // Safety check
            output[pos] = input[idx];
        }
    }
}

// __global__ void filterStringColumnKernel(const char **input, char **output, const bool *mask, const size_t row_count, unsigned long long *filtered_count)
// {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= row_count)
//         return;
//     if (mask[idx])
//     {
//         printf("idx: %zu, mask[idx]: %d\n", idx, mask[idx]);
//         printf("input[idx]: %s\n", input[idx]);
//         printf("output: %p\n", output);
//         printf("filtered_count: %llu\n", *filtered_count);
//         size_t pos = atomicAdd(filtered_count, 1ull);
//         if (pos < row_count)
//         {
//             output[pos] = const_cast<char *>(input[idx]);
//         }
//     }
// }

__global__ void computeOutputPositions(const bool* mask, unsigned int* positions, size_t size) {
    extern __shared__ unsigned int temp[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;
    
    temp[tid] = (idx < size && mask[idx]) ? 1 : 0;
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid >= s) {
            temp[tid] += temp[tid - s];
        }
        __syncthreads();
    }
    
    if (idx < size) {
        positions[idx] = temp[tid];
    }
    
    if (blockIdx.x > 0 && tid == 0) {
        unsigned int prefix = 0;
        for (int i = 0; i < blockIdx.x; i++) {
            prefix += positions[(i + 1) * blockDim.x - 1];
        }
        
        for (int i = 0; i < blockDim.x && (blockIdx.x * blockDim.x + i) < size; i++) {
            atomicAdd(&positions[blockIdx.x * blockDim.x + i], prefix);
        }
    }
}

template <typename T>
__global__ void copySelectedRowsKernel(const T *input, T *output,
                                       const bool *mask, const unsigned int *positions,
                                       size_t size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && mask[idx])
    {
        output[positions[idx] - 1] = input[idx];
    }
}

__global__ void copySelectedStringRowsKernel(const char **input, const char **output,
                                             const bool *mask, const unsigned int *positions,
                                             size_t size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && mask[idx])
    {
        output[positions[idx] - 1] = input[idx];
    }
}

template __global__ void filterKernel<int>(const int *, bool *, size_t, int, uint8_t);
template __global__ void filterKernel<float>(const float *, bool *, size_t, float, uint8_t);
template __global__ void filterKernel<int64_t>(const int64_t *, bool *, size_t, int64_t, uint8_t);

template __global__ void copySelectedRowsKernel<int>(const int *, int *, const bool *, const unsigned int *, const size_t);
template __global__ void copySelectedRowsKernel<float>(const float *, float *, const bool *, const unsigned int *, const size_t);
template __global__ void copySelectedRowsKernel<int64_t>(const int64_t *, int64_t *, const bool *, const unsigned int *, const size_t);