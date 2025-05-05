#include "kernels/aggregate/sum_kernel.hpp"

__global__ void findSumElement(float *input, float *output, int size)
{
    extern __shared__ float warp_sums[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Handle empty input
    if (size <= 0 && tid == 0)
    {
        *output = 0.0f;
        return;
    }

    float local_sum = 0.0f;

    // Load input data
    if (tid < size)
    {
        local_sum = input[tid];
    }

    // Warp-level reduction with active threads only
    unsigned active_mask = __ballot_sync(0xffffffff, tid < size);
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float neighbor = __shfl_down_sync(active_mask, local_sum, offset);
        if (tid + offset < size)
        {
            local_sum += neighbor;
        }
    }

    // Store warp result in shared memory
    if (lane_id == 0)
    {
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0 && lane_id < num_warps && lane_id * 32 < size)
    {
        local_sum = warp_sums[lane_id];

        unsigned reduction_mask = __ballot_sync(0xffffffff, lane_id < num_warps && lane_id * 32 < size);
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float neighbor = __shfl_down_sync(reduction_mask, local_sum, offset);
            if (lane_id + offset < num_warps && (lane_id + offset) * 32 < size)
            {
                local_sum += neighbor;
            }
        }

        if (lane_id == 0)
        {
            atomicAdd(output, local_sum);
        }
    }
}
