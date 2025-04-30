#include "kernels/filter_kernel.hpp"

// template <typename T>
__global__ void filterKernel(void *input, bool *output, size_t row_count, float value, uint8_t cond)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count)
        return;

    // T *typed_input = static_cast<T *>(input);
    // T val = typed_input[idx];
    float *typed_input = static_cast<float *>(input);
    float val = typed_input[idx];

    switch (cond)
    {
    case 1: // >
        output[idx] = (val > value);
        break;
    case 2: // <
        output[idx] = (val < value);
        break;
    case 3: // ==
        output[idx] = (val == value);
        break;
    case 4: // !=
        output[idx] = (val != value);
        break;
    case 5: // <=
        output[idx] = (val <= value);
        break;
    case 6: // >=
        output[idx] = (val >= value);
        break;
    default:
        output[idx] = false;
        break;
    }
}