// cuda_filters.cu
#include "kernels/filter_kernel.hpp"

__global__ void filterKernel(
    double *data,      // Flattened numeric data
    bool *output_mask, // Output boolean mask for filtered rows
    size_t row_count,  // Number of rows
    size_t col_idx,    // Column to filter
    double threshold,  // Filter threshold
    uint8_t op         // Operator: 1 ('>'), 2 ('<'), 3 ('=')
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < row_count)
    {
        double value = data[idx * gridDim.y + col_idx];
        bool result = false;

        switch (op)
        {
        case 1: // '>'
            result = value > threshold;
            break;
        case 2: // '<'
            result = value < threshold;
            break;
        case 3: // '='
            result = abs(value - threshold) < 1e-6;
            break;
        }

        output_mask[idx] = result;
    }
}
