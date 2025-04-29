#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"

// enum FilterOp
// {
//     OP_EQUAL,
//     OP_NOT_EQUAL,
//     OP_GREATER,
//     OP_LESS,
//     OP_GREATER_EQUAL,
//     OP_LESS_EQUAL,
//     OP_CONTAINS
// };

// struct FilterCondition
// {
//     size_t column_index;
//     FilterOp operation;
//     std::variant<float, int, int64_t, std::string> value;
// };

__global__ void filterKernel(
    double *data,      // Flattened numeric data
    bool *output_mask, // Output boolean mask for filtered rows
    size_t row_count,  // Number of rows
    size_t col_idx,    // Column to filter
    double threshold,  // Filter threshold
    uint8_t op         // Operator: 1 ('>'), 2 ('<'), 3 ('=')
);