#include "kernels/aggregate_kernel.hpp"

__device__ float atomicMaxFloat(float *address, float val)
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

__global__ void findMaxElement(float *input, float *output, int size)
{
    __shared__ float warp_maxes[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float local_max = -INFINITY;
    if (tid < size)
    {
        local_max = input[tid];
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, neighbor);
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
            float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, neighbor);
        }

        // First thread writes final result to global memory
        if (lane_id == 0)
        {
            atomicMaxFloat(output, local_max);
        }
    }
}
