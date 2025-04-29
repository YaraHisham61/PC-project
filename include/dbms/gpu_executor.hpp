// #pragma once
// #include <string>
// #include <vector>
// #include <variant>
// #include <cuda_runtime.h>
// #include "constants/db.hpp"
// #include "kernels/filter_kernel.hpp"
// #include <stdexcept>

// class GPUExecutor
// {
// public:
//     cudaError_t filterTable(
//         const TableResults &table,
//         size_t column_index,
//         double threshold,
//         char op, // 1 for '>', 2 for '<', 3 for '='
//         std::vector<bool> &result);
// };
