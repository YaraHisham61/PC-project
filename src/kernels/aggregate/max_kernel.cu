#include "kernels/aggregate/max_kernel.hpp"

template <typename T>
__device__ T atomicMaxGeneric(T *address, T val)
{
    static_assert(std::is_same<T, float>::value ||
                      std::is_same<T, int>::value ||
                      std::is_same<T, uint64_t>::value,
                  "Unsupported type for atomicMaxGeneric");

    if constexpr (std::is_same<T, float>::value)
    {
        int *address_as_int = (int *)address;
        int old = *address_as_int, assumed;
        do
        {
            assumed = old;
            if (__int_as_float(assumed) >= val)
                break;
            old = atomicCAS(address_as_int, assumed, __float_as_int(val));
        } while (assumed != old);
        return __int_as_float(old);
    }
    else if constexpr (std::is_same<T, int>::value)
    {
        int old = *address, assumed;
        do
        {
            assumed = old;
            if (assumed >= val)
                break;
            old = atomicCAS(address, assumed, val);
        } while (assumed != old);
        return old;
    }
    else if constexpr (std::is_same<T, uint64_t>::value)
    {
        uint64_t old = *address, assumed;
        do
        {
            assumed = old;
            if (assumed >= val)
                break;
            old = atomicCAS(reinterpret_cast<unsigned long long *>(address),
                            assumed,
                            val);
        } while (assumed != old);
        return old;
    }
}

__device__ int cuda_strcmp(const char *s1, const char *s2)
{
    while (*s1 && (*s1 == *s2))
    {
        s1++;
        s2++;
    }
    return (unsigned char)(*s1) - (unsigned char)(*s2);
}

__device__ char *atomicMaxString(char **address, char *new_ptr, const char **strings)
{
    char *old = *address, *assumed;
    do
    {
        assumed = old;
        // Compare strings pointed to by old and new_ptr
        if (old != nullptr && new_ptr != nullptr && cuda_strcmp(old, new_ptr) >= 0)
        {
            break; // old string is lexicographically >= new string
        }
        old = reinterpret_cast<char *>(atomicCAS(
            reinterpret_cast<unsigned long long *>(address),
            reinterpret_cast<unsigned long long>(assumed),
            reinterpret_cast<unsigned long long>(new_ptr)));
    } while (assumed != old);
    return old;
}
template <typename T>
__global__ void findMaxElement(T *input, T *output, int size)
{
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T *warp_maxes = reinterpret_cast<T *>(shared_mem);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    T local_max;
    if constexpr (std::is_same<T, float>::value)
    {
        local_max = -FLT_MAX;
    }

    else if constexpr (std::is_same<T, uint64_t>::value)
    {
        local_max = 0; // Minimum for unsigned is 0
    }

    if (tid < size)
    {
        local_max = input[tid];
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        T neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = max(local_max, neighbor);
    }

    if (lane_id == 0)
    {
        warp_maxes[warp_id] = local_max;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32)
    {
        local_max = warp_maxes[lane_id];

        for (int offset = 16; offset > 0; offset /= 2)
        {
            T neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = max(local_max, neighbor);
        }

        if (lane_id == 0)
        {
            atomicMaxGeneric(output, local_max);
        }
    }
}

template <>
__global__ void findMaxElement<char *>(char **input, char **output, int size)
{

    extern __shared__ char *warp_maxes[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    char *local_max = nullptr; // Initialize to nullptr for invalid threads

    if (tid < size)
    {
        local_max = input[tid];
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2)
    {
        // Cast char* to unsigned long long for shuffle
        unsigned long long local_max_ull = reinterpret_cast<unsigned long long>(local_max);
        unsigned long long neighbor_ull = __shfl_down_sync(0xffffffff, local_max_ull, offset);
        char *neighbor = reinterpret_cast<char *>(neighbor_ull);
        if (neighbor != nullptr)
        {
            if (local_max == nullptr || cuda_strcmp(local_max, neighbor) < 0)
            {
                local_max = neighbor;
            }
        }
    }

    // First thread in each warp writes to shared memory
    if (lane_id == 0)
    {
        warp_maxes[warp_id] = local_max;
    }

    __syncthreads();

    // First warp performs final reduction of all warps
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32)
    {
        local_max = warp_maxes[lane_id];

        for (int offset = 16; offset > 0; offset /= 2)
        {
            // Cast char* to unsigned long long for shuffle
            unsigned long long local_max_ull = reinterpret_cast<unsigned long long>(local_max);
            unsigned long long neighbor_ull = __shfl_down_sync(0xffffffff, local_max_ull, offset);
            char *neighbor = reinterpret_cast<char *>(neighbor_ull);
            if (neighbor != nullptr)
            {
                if (local_max == nullptr || cuda_strcmp(local_max, neighbor) < 0)
                {
                    local_max = neighbor;
                }
            }
        }

        // First thread writes final result to global memory
        if (lane_id == 0)
        {
            atomicMaxString(output, local_max, input);
        }
    }
}
template __device__ float atomicMaxGeneric<float>(float *, float);
template __device__ uint64_t atomicMaxGeneric<uint64_t>(uint64_t *, uint64_t);

template __global__ void findMaxElement<float>(float *, float *, int);
template __global__ void findMaxElement<uint64_t>(uint64_t *, uint64_t *, int);
template __global__ void findMaxElement<char *>(char **, char **, int);