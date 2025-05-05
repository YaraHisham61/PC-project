#include "kernels/aggregate/count_star_kernel.hpp"
__device__ float atomicAddFloat(float *address, float val)
{
    return atomicAdd(address, val);
}

template <typename T>
__global__ void countStar(T *input, float *output, int size)
{
    extern __shared__ __align__(sizeof(uint64_t)) unsigned char shared_mem[];
    uint64_t *warp_counts = reinterpret_cast<uint64_t *>(shared_mem);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    uint64_t local_count = 0;

    if (tid < size)
    {
        local_count = 1;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
        uint64_t neighbor = __shfl_down_sync(0xffffffff, local_count, offset);
        local_count += neighbor;
    }

    if (lane_id == 0)
    {
        warp_counts[warp_id] = local_count;
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32)
    {
        local_count = warp_counts[lane_id];

        for (int offset = 16; offset > 0; offset /= 2)
        {
            uint64_t neighbor = __shfl_down_sync(0xffffffff, local_count, offset);
            local_count += neighbor;
        }

        if (lane_id == 0)
        {
            atomicAdd(output, static_cast<float>(local_count));
        }
    }
}

template __global__ void countStar<float>(float *, float *, int);
template __global__ void countStar<uint64_t>(uint64_t *, float *, int);
template __global__ void countStar<char *>(char **, float *, int);
