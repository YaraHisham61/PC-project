// cuda_filters.cu
#include "kernels/filter_kernel.hpp"

template <typename T>
__global__ void filterKernel(
    T *data_in,
    bool *output_mask,
    size_t row_count,
    int threshold,
    uint8_t cond // 1: >, 2: <, 3: ==, 4: !=, 5: >=, 6: <=
)

{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < row_count)
    {
        bool result = false;
        switch (cond)
        {
        case 1:
            result = data_in[idx] > threshold;
            break;
        case 2:
            result = data_in[idx] < threshold;
            break;
        case 3: 
            result = data_in[idx] == threshold;
            break;
        case 4:
            result = data_in[idx] != threshold;
            break;
        case 5:
            result = data_in[idx] >= threshold;
            break;
        case 6:
            result = data_in[idx] <= threshold;
            break;
        default:
            break;
        }
        output_mask[idx] = result;
    }
}
