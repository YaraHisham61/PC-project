#include "kernels/hash_join.hpp"

__device__ int device_strcmp22(const char *s1, const char *s2)
{
    while (*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

// Primary template
template <typename T>
__global__ void hashJoinKernel(
    const T *left_data, const T *right_data,
    bool *output_mask_left, bool *output_mask_right,
    size_t left_size, size_t right_size)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T *shared_right = reinterpret_cast<T *>(shared_mem);

    // Each block processes a chunk of the right table
    for (size_t block_start = 0; block_start < right_size; block_start += blockDim.x)
    {
        size_t right_idx = block_start + threadIdx.x;

        // Load right table into shared memory
        if (right_idx < right_size)
        {
            shared_right[threadIdx.x] = right_data[right_idx];
        }
        __syncthreads();

        // Process left table elements
        size_t left_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (left_idx < left_size)
        {
            T left_val = left_data[left_idx];

            // Compare with right table in shared memory
            size_t limit = min(static_cast<size_t>(blockDim.x), right_size - block_start);
            for (size_t i = 0; i < limit; i++)
            {
                if (left_val == shared_right[i])
                {
                    output_mask_left[left_idx] = true;
                    output_mask_right[block_start + i] = true;
                    // break;
                }
            }
        }
        __syncthreads();
    }
}

// Specialization for string pointers
template <>
__global__ void hashJoinKernel<const char *>(
    const char *const *left_data, const char *const *right_data,
    bool *output_mask_left, bool *output_mask_right,
    size_t left_size, size_t right_size)
{
    extern __shared__ __align__(sizeof(const char *)) unsigned char shared_mem[];
    const char **shared_right = reinterpret_cast<const char **>(shared_mem);

    for (size_t block_start = 0; block_start < right_size; block_start += blockDim.x)
    {
        size_t right_idx = block_start + threadIdx.x;

        // Load right table pointers into shared memory
        if (right_idx < right_size)
        {
            shared_right[threadIdx.x] = right_data[right_idx];
        }
        __syncthreads();

        // Process left table elements
        size_t left_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (left_idx < left_size)
        {
            const char *left_str = left_data[left_idx];

            // Compare with right table in shared memory
            size_t limit = min(static_cast<size_t>(blockDim.x), right_size - block_start);
            for (size_t i = 0; i < limit; i++)
            {
                if (device_strcmp22(left_str, shared_right[i]) == 0)
                {
                    output_mask_left[left_idx] = true;
                    output_mask_right[block_start + i] = true;
                    // break;
                }
            }
        }
        __syncthreads();
    }
}

// Explicit instantiations
template __global__ void hashJoinKernel<float>(
    const float *, const float *,
    bool *, bool *,
    size_t, size_t);

template __global__ void hashJoinKernel<uint64_t>(
    const uint64_t *, const uint64_t *,
    bool *, bool *,
    size_t, size_t);

template __global__ void hashJoinKernel<const char *>(
    const char *const *, const char *const *,
    bool *, bool *,
    size_t, size_t);