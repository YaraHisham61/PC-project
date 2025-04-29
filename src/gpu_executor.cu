// #include "dbms/gpu_executor.hpp"
// #include "kernels/filter_kernel.hpp"
// #include <stdexcept>

// cudaError_t GPUExecutor::filterTable(
//     const TableResults &table,
//     size_t column_index,
//     double threshold,
//     char op, // 1 for '>', 2 for '<', 3 for '='
//     std::vector<bool> &result)
// {
//     if (column_index >= table.column_count)
//     {
//         throw std::runtime_error("Invalid column index");
//     }

//     // Check if column is numeric
//     DataType col_type = table.columns[column_index].type;
//     if (col_type != DataType::INT && col_type != DataType::FLOAT)
//     {
//         throw std::runtime_error("Filter only supported for INT and FLOAT columns");
//     }

//     // Allocate and copy data to GPU
//     double *d_data;
//     bool *d_mask;
//     size_t row_count = table.row_count;

//     // Flatten data to double array
//     std::vector<double> h_data(row_count * table.column_count);
//     for (size_t i = 0; i < row_count; ++i)
//     {
//         for (size_t j = 0; j < table.column_count; ++j)
//         {
//             size_t idx = i * table.column_count + j;
//             auto &value = table.rows[idx];

//             if (table.columns[j].type == DataType::FLOAT)
//             {
//                 h_data[idx] = std::get<float>(value);
//             }
//             else if (table.columns[j].type == DataType::INT)
//             {
//                 h_data[idx] = std::get<int>(value);
//             }
//             else if (table.columns[j].type == DataType::DATETIME)
//             {
//                 h_data[idx] = static_cast<double>(std::get<int64_t>(value));
//             }
//             else
//             {
//                 h_data[idx] = 0.0; // STRING columns are not filtered
//             }
//         }
//     }

//     // Allocate GPU memory
//     cudaError_t cudaStatus;
//     cudaStatus = cudaMalloc(&d_data, row_count * table.column_count * sizeof(double));
//     if (cudaStatus != cudaSuccess)
//         return cudaStatus;

//     cudaStatus = cudaMalloc(&d_mask, row_count * sizeof(bool));
//     if (cudaStatus != cudaSuccess)
//     {
//         cudaFree(d_data);
//         return cudaStatus;
//     }

//     // Copy data to GPU
//     cudaStatus = cudaMemcpy(d_data, h_data.data(),
//                             row_count * table.column_count * sizeof(double),
//                             cudaMemcpyHostToDevice);
//     if (cudaStatus != cudaSuccess)
//     {
//         cudaFree(d_data);
//         cudaFree(d_mask);
//         return cudaStatus;
//     }

//     // Launch kernel
//     int threadsPerBlock = 256;
//     int blocksPerGrid = (row_count + threadsPerBlock - 1) / threadsPerBlock;

//     filterKernel<<<blocksPerGrid, threadsPerBlock>>>(
//         d_data, d_mask, row_count, column_index, threshold, static_cast<uint8_t>(op));

//     cudaStatus = cudaGetLastError();
//     if (cudaStatus != cudaSuccess)
//     {
//         cudaFree(d_data);
//         cudaFree(d_mask);
//         return cudaStatus;
//     }

//     // Copy results back
//     // Use a temporary vector<char> for the transfer since std::vector<bool> doesn't have .data()
//     std::vector<char> temp_result(row_count);
//     cudaStatus = cudaMemcpy(temp_result.data(), d_mask,
//                             row_count * sizeof(bool),
//                             cudaMemcpyDeviceToHost);
//     if (cudaStatus != cudaSuccess)
//     {
//         cudaFree(d_data);
//         cudaFree(d_mask);
//         return cudaStatus;
//     }

//     // Convert temp_result (char) to result (bool)
//     result.resize(row_count);
//     for (size_t i = 0; i < row_count; ++i)
//     {
//         result[i] = temp_result[i] != 0;
//     }

//     // Clean up
//     cudaFree(d_data);
//     cudaFree(d_mask);

//     return cudaStatus;
// }