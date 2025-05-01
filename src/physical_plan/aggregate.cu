#include "physical_plan/aggregate.hpp"

Aggregate::Aggregate(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("Aggregates");
    if (it != params.end())
    {
        parseAggregateList(it->second);
    }
}

void Aggregate::parseAggregateList(const std::string &aggregate_list)
{
    std::istringstream iss(aggregate_list);
    std::string agg_str;

    while (std::getline(iss, agg_str, '\n'))
    {
        agg_str.erase(0, agg_str.find_first_not_of(" \t"));
        agg_str.erase(agg_str.find_last_not_of(" \t") + 1);

        if (!agg_str.empty())
        {
            aggregates.push_back(parseSingleAggregate(agg_str));
        }
    }
}

AggregateFunction Aggregate::parseSingleAggregate(const std::string &agg_str) const
{
    std::istringstream iss(agg_str);
    std::string func_name;
    iss >> func_name;

    AggregateFunction func;

    if (func_name == "count_star()")
    {
        func.type = AggregateType::COUNT_STAR;
        func.column_index = -1;
    }
    else if (func_name.find("count(#") == 0)
    {
        func.type = AggregateType::COUNT;
        func.column_index = std::stoi(agg_str.substr(7, agg_str.find(')') - 7));
    }
    else if (func_name.find("avg(#") == 0)
    {
        func.type = AggregateType::AVG;
        func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
    }
    else if (func_name.find("min(#") == 0)
    {
        func.type = AggregateType::MIN;
        func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
    }
    else if (func_name.find("max(#") == 0)
    {
        func.type = AggregateType::MAX;
        func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
    }
    else if (func_name.find("sum(#") == 0)
    {
        func.type = AggregateType::SUM;
        func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
    }
    else
    {
        throw std::runtime_error("Unknown aggregate function: " + agg_str);
    }

    return func;
}

TableResults Aggregate::computeAggregates(const TableResults &input) const
{
    TableResults result;
    result.row_count = 1;
    result.column_count = this->aggregates.size();
    result.data.resize(input.column_count);

    int numThreads = 256;
    int numBlocks = (input.row_count + numThreads - 1) / numThreads;

    for (size_t i = 0; i < aggregates.size(); ++i)
    {
        size_t shared_mem_size = ((numThreads + 31) / 32) * getDataTypeNumBytes(input.columns[i].type);
        ColumnInfo col;
        col.name = getAggregateName(aggregates[i], input);
        col.type = getOutputType(aggregates[i], input);
        col.idx = aggregates[i].column_index;
        result.columns.push_back(col);

        switch (aggregates[i].type)
        {
        case AggregateType::COUNT_STAR:
            break;
        case AggregateType::COUNT:
            break;
        case AggregateType::SUM:
            break;
        case AggregateType::AVG:
            break;
        case AggregateType::MIN:
            break;
        case AggregateType::MAX:
            switch (input.columns[i].type)
            {
            case DataType::FLOAT:
            {
                float *d_input_f = nullptr, *d_output_f = nullptr;
                cudaMalloc(&d_input_f, input.row_count * sizeof(float));
                cudaMalloc(&d_output_f, sizeof(float));
                cudaMemcpy(d_input_f, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(float), cudaMemcpyHostToDevice);

                findMaxElement<float><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_f, d_output_f, input.row_count);

                float f_value;
                cudaDeviceSynchronize();
                cudaMemcpy(&f_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                result.data[i] = new float(f_value);

                cudaFree(d_input_f);
                cudaFree(d_output_f);
                break;
            }
            case DataType::INT:
            {
                int *d_input_i = nullptr, *d_output_i = nullptr;
                cudaMalloc(&d_input_i, input.row_count * sizeof(int));
                cudaMalloc(&d_output_i, sizeof(int));
                cudaMemcpy(d_input_i, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(int), cudaMemcpyHostToDevice);

                findMaxElement<int><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_i, d_output_i, input.row_count);

                int int_value;
                cudaMemcpy(&int_value, d_output_i, sizeof(int), cudaMemcpyDeviceToHost);
                result.data[i] = new int(int_value);

                cudaFree(d_input_i);
                cudaFree(d_output_i);
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t *d_input_ui = nullptr, *d_output_ui = nullptr;
                cudaMalloc(&d_input_ui, input.row_count * sizeof(uint64_t));
                cudaMalloc(&d_output_ui, sizeof(uint64_t));
                cudaMemcpy(d_input_ui, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

                findMaxElement<uint64_t><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_ui, d_output_ui, input.row_count);

                uint64_t datetime_value;
                cudaMemcpy(&datetime_value, d_output_ui, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                result.data[i] = new uint64_t(datetime_value);

                cudaFree(d_input_ui);
                cudaFree(d_output_ui);
                break;
            }
            case DataType::STRING:
            {
                // const char **d_col_data = nullptr;
                // cudaMalloc(&d_col_data, row_count * sizeof(char *));
                // const char **host_strings = static_cast<const char **>(input_table.data[col_idx]);

                // char **d_strings = new char *[row_count];

                // for (size_t i = 0; i < row_count; i++)
                // {
                //     size_t len = strlen(host_strings[i]) + 1;
                //     cudaMalloc(&d_strings[i], len);
                //     cudaMemcpy(d_strings[i], host_strings[i], len, cudaMemcpyHostToDevice);
                //     cudaMemcpy(&d_col_data[i], &d_strings[i], sizeof(char *), cudaMemcpyHostToDevice);
                // }

                // char *d_value = nullptr;
                // cudaMalloc(&d_value, cond.value.size() + 1);
                // cudaMemcpy(d_value, cond.value.c_str(), cond.value.size() + 1, cudaMemcpyHostToDevice);
                // filterKernelString<<<numBlocks, numThreads>>>(d_col_data, d_temp_mask, row_count, d_value, cond_code);

                /////////////////////////////
                char **d_input_char = nullptr, **d_output_char = nullptr;
                cudaMalloc(&d_input_char, input.row_count * sizeof(char *));
                cudaMalloc(&d_output_char, sizeof(char *));

                // Allocate device memory for string pointers
                char **d_strings = new char *[input.row_count];
                const char **host_strings = static_cast<const char **>(input.data[aggregates[i].column_index]);

                for (size_t j = 0; j < input.row_count; j++)
                {
                    size_t len = strlen(host_strings[j]) + 1; // Include null terminator
                    cudaMalloc(&d_strings[j], len);
                    cudaMemcpy(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice);
                    cudaMemcpy(&d_input_char[j], &d_strings[j], sizeof(char *), cudaMemcpyHostToDevice);
                }

                // Launch the kernel to find the maximum element
                findMaxElement<char *><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_char, d_output_char, input.row_count);
                cudaDeviceSynchronize();

                // Copy the output pointer (char*) from device to host
                char *d_max_string = nullptr;
                cudaMemcpy(&d_max_string, d_output_char, sizeof(char *), cudaMemcpyDeviceToHost);

                // Determine the length of the output string
                size_t max_len = 0;
                if (d_max_string)
                {
                    // Use a small kernel or cudaMemcpy to get the string length (or assume a max length)
                    // For simplicity, assume a maximum length or query the length from the device
                    // Here, we copy the string to host to get its length
                    char temp_buffer[256]; // Temporary buffer (adjust size as needed)
                    cudaMemcpy(temp_buffer, d_max_string, sizeof(temp_buffer), cudaMemcpyDeviceToHost);
                    max_len = strlen(temp_buffer) + 1; // Include null terminator
                }

                // Allocate host memory for the result string
                char *h_output_string = new char[max_len];
                cudaMemcpy(h_output_string, d_max_string, max_len, cudaMemcpyDeviceToHost);
                std::cout << "Max string: " << h_output_string << std::endl;

                result.data[i] = new char *[1];
                static_cast<char **>(result.data[i])[0] = h_output_string;
                std::cout << "Max string: " << static_cast<char **>(result.data[i])[0] << std::endl;
                result.print();
                // Cleanup
                for (size_t j = 0; j < input.row_count; j++)
                {
                    cudaFree(d_strings[j]);
                }
                delete[] d_strings;
                cudaFree(d_input_char);
                cudaFree(d_output_char);
                delete[] h_output_string;

                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for MAX aggregate");
            }
            break;
        }
    }

    return result;
}

// // // Helper function to check for null values
// // bool Aggregate::isNull(const ValueVariant &val) const
// // {
// //     return val.valueless_by_exception();
// // }

std::string Aggregate::getAggregateName(const AggregateFunction &agg, const TableResults &input) const
{
    switch (agg.type)
    {
    case AggregateType::COUNT_STAR:
        return "count_star";
    case AggregateType::COUNT:
        return "count_" + input.columns[agg.column_index].name;
    case AggregateType::SUM:
        return "sum_" + input.columns[agg.column_index].name;
    case AggregateType::AVG:
        return "avg_" + input.columns[agg.column_index].name;
    case AggregateType::MIN:
        return "min_" + input.columns[agg.column_index].name;
    case AggregateType::MAX:
        return "max_" + input.columns[agg.column_index].name;
    }
    return "agg";
}

DataType Aggregate::getOutputType(const AggregateFunction &agg, const TableResults &input) const
{
    switch (agg.type)
    {
    case AggregateType::COUNT_STAR:
    case AggregateType::COUNT:
        return DataType::INT;
    case AggregateType::SUM:
    case AggregateType::AVG:
        return DataType::FLOAT;
    case AggregateType::MIN:
    case AggregateType::MAX:
        return input.columns[agg.column_index].type;
    }
    return DataType::INT;
}

// // int64_t Aggregate::countNonNull(const TableResults &input, int col_idx) const
// // {
// //     int64_t count = 0;
// //     for (size_t row = 0; row < input.row_count; ++row)
// //     {
// //         size_t idx = row * input.column_count + col_idx;
// //         if (idx < input.rows.size() && !isNull(input.rows[idx]))
// //         {
// //             count++;
// //         }
// //     }
// //     return count;
// // }

// // float Aggregate::computeSum(const TableResults &input, int col_idx) const
// // {
// //     double sum = 0;
// //     for (size_t row = 0; row < input.row_count; ++row)
// //     {
// //         size_t idx = row * input.column_count + col_idx;
// //         if (idx < input.rows.size() && !isNull(input.rows[idx]))
// //         {
// //             if (std::holds_alternative<float>(input.rows[idx]))
// //             {
// //                 sum += std::get<float>(input.rows[idx]);
// //             }
// //             else if (std::holds_alternative<int>(input.rows[idx]))
// //             {
// //                 sum += std::get<int>(input.rows[idx]);
// //             }
// //             else if (std::holds_alternative<int64_t>(input.rows[idx]))
// //             {
// //                 sum += std::get<int64_t>(input.rows[idx]);
// //             }
// //         }
// //     }
// //     return static_cast<float>(sum);
// // }

// // float Aggregate::computeAvg(const TableResults &input, int col_idx) const
// // {
// //     double sum = 0;
// //     int64_t count = 0;
// //     for (size_t row = 0; row < input.row_count; ++row)
// //     {
// //         size_t idx = row * input.column_count + col_idx;
// //         if (idx < input.rows.size() && !isNull(input.rows[idx]))
// //         {
// //             if (std::holds_alternative<float>(input.rows[idx]))
// //             {
// //                 sum += std::get<float>(input.rows[idx]);
// //                 count++;
// //             }
// //             else if (std::holds_alternative<int>(input.rows[idx]))
// //             {
// //                 sum += std::get<int>(input.rows[idx]);
// //                 count++;
// //             }
// //             else if (std::holds_alternative<int64_t>(input.rows[idx]))
// //             {
// //                 sum += std::get<int64_t>(input.rows[idx]);
// //                 count++;
// //             }
// //         }
// //     }
// //     return count > 0 ? static_cast<float>(sum / count) : 0.0f;
// // }

// // ValueVariant Aggregate::findMin(const TableResults &input, int col_idx) const
// // {
// //     ValueVariant min_val;
// //     bool has_value = false;

// //     for (size_t row = 0; row < input.row_count; ++row)
// //     {
// //         size_t idx = row * input.column_count + col_idx;
// //         if (idx >= input.rows.size() || isNull(input.rows[idx]))
// //             continue;

// //         const auto &val = input.rows[idx];
// //         if (!has_value)
// //         {
// //             min_val = val;
// //             has_value = true;
// //         }
// //         else
// //         {
// //             if (val < min_val)
// //             {
// //                 min_val = val;
// //             }
// //         }
// //     }

// //     if (!has_value)
// //     {
// //         // Return default value based on column type
// //         DataType type = input.columns[col_idx].type;
// //         switch (type)
// //         {
// //         case DataType::FLOAT:
// //             return 0.0f;
// //         case DataType::INT:
// //             return 0;
// //         case DataType::STRING:
// //             return std::string();
// //         default:
// //             return 0.0f;
// //         }
// //     }
// //     return min_val;
// // }

void Aggregate::print() const
{
    std::cout << "UNGROUPED_AGGREGATE (";
    for (size_t i = 0; i < aggregates.size(); ++i)
    {
        if (i > 0)
            std::cout << ", ";

        switch (aggregates[i].type)
        {
        case AggregateType::COUNT_STAR:
            std::cout << "count_star()";
            break;
        case AggregateType::COUNT:
            std::cout << "count(#" << aggregates[i].column_index << ")";
            break;
        case AggregateType::SUM:
            std::cout << "sum(#" << aggregates[i].column_index << ")";
            break;
        case AggregateType::AVG:
            std::cout << "avg(#" << aggregates[i].column_index << ")";
            break;
        case AggregateType::MIN:
            std::cout << "min(#" << aggregates[i].column_index << ")";
            break;
        case AggregateType::MAX:
            std::cout << "max(#" << aggregates[i].column_index << ")";
            break;
        }

        if (!output_names.empty() && i < output_names.size())
        {
            std::cout << " AS " << output_names[i];
        }
    }
    std::cout << ")\n";
}

// case DataType::STRING:
// {
//     char** d_input_char = nullptr, **d_output_char = nullptr;
//     cudaMalloc(&d_input_char, input.row_count * sizeof(char*));
//     cudaMalloc(&d_output_char, sizeof(char*));

//     // Allocate device memory for string pointers
//     char** d_strings = new char*[input.row_count];
//     const char** host_strings = static_cast<const char**>(input.data[aggregates[i].column_index]);

//     // First pass: find the maximum string length
//     size_t max_len = 0;
//     for (size_t j = 0; j < input.row_count; j++) {
//         size_t len = strlen(host_strings[j]) + 1; // Include null terminator
//         if (len > max_len) max_len = len;
//     }

//     // Second pass: copy all strings with proper allocation
//     for (size_t j = 0; j < input.row_count; j++) {
//         size_t len = strlen(host_strings[j]) + 1;
//         cudaMalloc(&d_strings[j], max_len); // Allocate uniform size
//         cudaMemcpy(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice);
//         cudaMemcpy(&d_input_char[j], &d_strings[j], sizeof(char*), cudaMemcpyHostToDevice);
//     }

//     // Launch the kernel to find the maximum element
//     findMaxElement<char*><<<numBlocks, numThreads, shared_mem_size>>>(
//         d_input_char, d_output_char, input.row_count);
//     cudaDeviceSynchronize();

//     // Copy the output pointer (char*) from device to host
//     char* d_max_string = nullptr;
//     cudaMemcpy(&d_max_string, d_output_char, sizeof(char*), cudaMemcpyDeviceToHost);

//     // Allocate host memory for the result string
//     char* h_output_string = new char[max_len];
//     cudaMemcpy(h_output_string, d_max_string, max_len, cudaMemcpyDeviceToHost);

//     // Store result
//     result.data[i] = new char[strlen(h_output_string) + 1];
//     strcpy(static_cast<char*>(result.data[i]), h_output_string);
//     std::cout << "Max string: " << static_cast<char*>(result.data[i]) << std::endl;

//     // Cleanup
//     for (size_t j = 0; j < input.row_count; j++) {
//         cudaFree(d_strings[j]);
//     }
//     delete[] d_strings;
//     cudaFree(d_input_char);
//     cudaFree(d_output_char);
//     delete[] h_output_string;

//     break;
// }

// case DataType::STRING:
// {
//     char** d_input_char = nullptr;
//     char** d_output_char = nullptr;
//     cudaMalloc(&d_input_char, input.row_count * sizeof(char*));
//     cudaMalloc(&d_output_char, sizeof(char*));

//     // Allocate device memory for string pointers
//     char** d_strings = new char*[input.row_count];
//     const char** host_strings = static_cast<const char**>(input.data[aggregates[i].column_index]);

//     // Copy each string to device memory
//     for (size_t j = 0; j < input.row_count; j++) {
//         size_t len = strlen(host_strings[j]) + 1; // Include null terminator
//         cudaMalloc(&d_strings[j], len);
//         cudaMemcpy(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice);
//         cudaMemcpy(&d_input_char[j], &d_strings[j], sizeof(char*), cudaMemcpyHostToDevice);
//     }

//     // Launch the kernel to find the maximum element
//     findMaxElement<char*><<<numBlocks, numThreads, shared_mem_size>>>(
//         d_input_char, d_output_char, input.row_count);
//     cudaDeviceSynchronize();

//     // Copy the output pointer (char*) from device to host
//     char* d_max_string = nullptr;
//     cudaMemcpy(&d_max_string, d_output_char, sizeof(char*), cudaMemcpyDeviceToHost);

//     // Get the actual string length from device
//     size_t max_len = 0;
//     if (d_max_string) {
//         // First copy just the length (4 bytes should be enough for length prefix)
//         uint32_t len_on_device = 0;
//         cudaMemcpy(&len_on_device, d_max_string, sizeof(uint32_t), cudaMemcpyDeviceToHost);

//         // If no length prefix, fall back to copying in chunks
//         if (len_on_device > 1000000) { // Unreasonable length indicates no prefix
//             const size_t chunk_size = 256;
//             char buffer[chunk_size];
//             size_t total_len = 0;

//             do {
//                 cudaMemcpy(buffer, d_max_string + total_len, chunk_size, cudaMemcpyDeviceToHost);
//                 size_t chunk_len = strnlen(buffer, chunk_size);
//                 total_len += chunk_len;
//                 if (chunk_len < chunk_size) break;
//             } while (true);

//             max_len = total_len + 1;
//         } else {
//             max_len = len_on_device + 1;
//         }
//     }

//     // Allocate host memory for the result string
//     char* h_output_string = new char[max_len];
//     cudaMemcpy(h_output_string, d_max_string, max_len, cudaMemcpyDeviceToHost);

//     // Store result
//     result.data[i] = new char[strlen(h_output_string) + 1];
//     strcpy(static_cast<char*>(result.data[i]), h_output_string);
//     std::cout << "Max string: " << static_cast<char*>(result.data[i]) << std::endl;

//     // Cleanup
//     for (size_t j = 0; j < input.row_count; j++) {
//         cudaFree(d_strings[j]);
//     }
//     delete[] d_strings;
//     cudaFree(d_input_char);
//     cudaFree(d_output_char);
//     delete[] h_output_string;

//     break;
// }