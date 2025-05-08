#include "kernels/hash_join_kernel.hpp"

__device__ int device_strcmp22(const char *s1, const char *s2)
{
    while (*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return *(const unsigned char *)s1 - *(const unsigned char *)s2;
}

template <typename T>
__global__ void hashJoinKernel(
    const T *__restrict__ left_data,
    const T *__restrict__ right_data,
    size_t left_size,
    size_t right_size,
    size_t *__restrict__ out_left_idx,
    size_t *__restrict__ out_right_idx,
    unsigned long long *__restrict__ out_count)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T *shared_right = reinterpret_cast<T *>(shared_mem);

    for (size_t block_start = 0; block_start < right_size; block_start += blockDim.x)
    {
        size_t right_idx = block_start + threadIdx.x;
        if (right_idx < right_size)
            shared_right[threadIdx.x] = right_data[right_idx];
        __syncthreads();

        size_t left_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (left_idx < left_size)
        {
            T left_val = left_data[left_idx];
            size_t limit = min(static_cast<size_t>(blockDim.x), right_size - block_start);
            for (size_t i = 0; i < limit; ++i)
            {
                if (left_val == shared_right[i])
                {
                    unsigned long long pos = atomicAdd(out_count, 1ULL);
                    out_left_idx[pos] = left_idx;
                    out_right_idx[pos] = block_start + i;
                }
            }
        }
        __syncthreads();
    }
}

template <>
__global__ void hashJoinKernel<const char *>(
    const char *const *__restrict__ left_data,
    const char *const *__restrict__ right_data,
    size_t left_size,
    size_t right_size,
    size_t *__restrict__ out_left_idx,
    size_t *__restrict__ out_right_idx,
    unsigned long long *__restrict__ out_count)
{
    extern __shared__ __align__(sizeof(const char *)) unsigned char shared_mem[];
    const char **shared_right = reinterpret_cast<const char **>(shared_mem);

    for (size_t block_start = 0; block_start < right_size; block_start += blockDim.x)
    {
        size_t right_idx = block_start + threadIdx.x;
        if (right_idx < right_size)
            shared_right[threadIdx.x] = right_data[right_idx];
        __syncthreads();

        size_t left_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (left_idx < left_size)
        {
            const char *left_str = left_data[left_idx];
            size_t limit = min(static_cast<size_t>(blockDim.x), right_size - block_start);
            for (size_t i = 0; i < limit; ++i)
            {
                if (device_strcmp22(left_str, shared_right[i]) == 0)
                {
                    unsigned long long pos = atomicAdd(out_count, 1ULL);
                    out_left_idx[pos] = left_idx;
                    out_right_idx[pos] = block_start + i;
                }
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void getRowsKernel(
    const T *__restrict__ in,
    const size_t *__restrict__ idx,
    T *__restrict__ out,
    size_t out_size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size)
    {
        out[i] = in[idx[i]];
    }
}

template <>
__global__ void getRowsKernel<const char *>(
    const char *const *__restrict__ in,
    const size_t *__restrict__ idx,
    const char **__restrict__ out,
    size_t out_size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_size)
    {
        out[i] = in[idx[i]];
    }
}

template __global__ void hashJoinKernel<float>(
    const float *, const float *,
    size_t, size_t,
    size_t *, size_t *, unsigned long long *);

template __global__ void hashJoinKernel<uint64_t>(
    const uint64_t *, const uint64_t *,
    size_t, size_t,
    size_t *, size_t *, unsigned long long *);

template __global__ void hashJoinKernel<const char *>(
    const char *const *, const char *const *,
    size_t, size_t,
    size_t *, size_t *, unsigned long long *);

template __global__ void getRowsKernel<float>(
    const float *, const size_t *,
    float *, size_t);
template __global__ void getRowsKernel<uint64_t>(
    const uint64_t *, const size_t *,
    uint64_t *, size_t);
template __global__ void getRowsKernel<const char *>(
    const char *const *, const size_t *,
    const char **, size_t);