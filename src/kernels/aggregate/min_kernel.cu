#include "kernels/aggregate/min_kernel.hpp"

template <typename T>
__device__ T atomicMinGeneric(T *address, T val)
{
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, int>::value ||
                      std::is_same<T, uint64_t>::value,
                  "Unsupported type for atomicMinGeneric");

    if constexpr (std::is_same<T, float>::value)
    {
        int *address_as_int = (int *)address;
        int old = *address_as_int, assumed;
        do
        {
            assumed = old;
            if (__int_as_float(assumed) <= val)
                break;
            old = atomicCAS(address_as_int, assumed, __float_as_int(val));
        } while (assumed != old);
        return __int_as_float(old);
    }
    else if constexpr (std::is_same<T, uint64_t>::value)
    {
        uint64_t old = *address, assumed;
        do
        {
            assumed = old;
            if (assumed <= val)
                break;
            old = atomicCAS(reinterpret_cast<unsigned long long *>(address),
                            assumed,
                            val);
        } while (assumed != old);
        return old;
    }
}

__device__ int cuda_strcmp_min(const char *s1, const char *s2)
{
    while (*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return (unsigned char)(*s1) - (unsigned char)(*s2);
}

__device__ char *atomicMinString(char **address, char *new_ptr, const char **strings)
{
    char *old = *address, *assumed;
    do
    {
        assumed = old;
        if (old != nullptr && new_ptr != nullptr && cuda_strcmp_min(old, new_ptr) <= 0)
        {
            break; // old string is lexicographically <= new string
        }
        old = reinterpret_cast<char *>(atomicCAS(
            reinterpret_cast<unsigned long long *>(address),
            reinterpret_cast<unsigned long long>(assumed),
            reinterpret_cast<unsigned long long>(new_ptr)));
    } while (assumed != old);
    return old;
}

template <typename T>
__global__ void findMinElement(T *input, T *output, int size)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T *warp_mins = reinterpret_cast<T *>(shared_mem);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Handle empty input
    if (size <= 0 && tid == 0)
    {
        *output = FLT_MAX;
        return;
    }

    T local_min;
    if constexpr (std::is_same<T, float>::value)
    {
        local_min = FLT_MAX;
    }
    else if constexpr (std::is_same<T, uint64_t>::value)
    {
        local_min = UINT64_MAX;
    }

    // Load input data
    if (tid < size)
    {
        local_min = input[tid];
    }

    // Warp-level reduction with active threads only
    unsigned active_mask = __ballot_sync(0xffffffff, tid < size);
    for (int offset = 16; offset > 0; offset /= 2)
    {
        T neighbor = __shfl_down_sync(active_mask, local_min, offset);
        if (tid + offset < size)
        {
            local_min = min(local_min, neighbor);
        }
    }

    if (lane_id == 0)
    {
        warp_mins[warp_id] = local_min;
    }

    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0 && lane_id < num_warps && lane_id * 32 < size)
    {
        local_min = warp_mins[lane_id];
        // printf("Before reduction, lane %d: %f\n", lane_id, local_min);

        unsigned reduction_mask = __ballot_sync(0xffffffff, lane_id < num_warps && lane_id * 32 < size);
        for (int offset = 16; offset > 0; offset /= 2)
        {
            // printf("Before Before reduction, lane %d: %f\n", lane_id, local_min);
            T neighbor = __shfl_down_sync(reduction_mask, local_min, offset);
            if (lane_id + offset < num_warps && (lane_id + offset) * 32 < size)
            {

                local_min = min(local_min, neighbor);
                // printf("After reduction, lane %d: %f\n", lane_id, local_min);
            }
        }

        if (lane_id == 0)
        {
            atomicMinGeneric(output, local_min);
        }
    }
}

template <>
__global__ void findMinElement<char *>(char **input, char **output, int size)
{
    extern __shared__ char *warp_mins[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    char *local_min = nullptr;

    if (tid < size)
    {
        local_min = input[tid];
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        unsigned long long local_min_ull = reinterpret_cast<unsigned long long>(local_min);
        unsigned long long neighbor_ull = __shfl_down_sync(0xffffffff, local_min_ull, offset);
        char *neighbor = reinterpret_cast<char *>(neighbor_ull);
        if (neighbor != nullptr)
        {
            if (local_min == nullptr || cuda_strcmp_min(local_min, neighbor) > 0)
            {
                local_min = neighbor;
            }
        }
    }

    if (lane_id == 0)
    {
        warp_mins[warp_id] = local_min;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32)
    {
        local_min = warp_mins[lane_id];

        for (int offset = 16; offset > 0; offset /= 2)
        {
            unsigned long long local_min_ull = reinterpret_cast<unsigned long long>(local_min);
            unsigned long long neighbor_ull = __shfl_down_sync(0xffffffff, local_min_ull, offset);
            char *neighbor = reinterpret_cast<char *>(neighbor_ull);
            if (neighbor != nullptr)
            {
                if (local_min == nullptr || cuda_strcmp_min(local_min, neighbor) > 0)
                {
                    local_min = neighbor;
                }
            }
        }

        if (lane_id == 0)
        {
            atomicMinString(output, local_min, input);
        }
    }
}

template __device__ float atomicMinGeneric<float>(float *, float);
template __device__ uint64_t atomicMinGeneric<uint64_t>(uint64_t *, uint64_t);

template __global__ void findMinElement<float>(float *, float *, int);
template __global__ void findMinElement<uint64_t>(uint64_t *, uint64_t *, int);
template __global__ void findMinElement<char *>(char **, char **, int);