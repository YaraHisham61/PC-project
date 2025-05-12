#include "physical_plan/nested_join.hpp"

NestedLoopJoin::NestedLoopJoin(const duckdb::InsertionOrderPreservingMap<std::string> &params)
    : PhysicalOpNode()
{
    auto it = params.find("Conditions");
    std::string column_name;
    if (it != params.end())
    {
        std::string condition = it->second;
        std::istringstream iss(condition);
        std::string keyword;
        iss >> keyword >> op >> column_name;
    }
    if (column_name.find("CAST(") != std::string::npos)
    {
        this->column = column_name.substr(5);
    }
    else
    {
        this->column = column_name;
    }

    std::cout << "NestedLoopJoin: " << column << " " << op << "\n";
}

std::string NestedLoopJoin::extract_base_column_name(std::string column_name)
{
    // Normalize by removing leading/trailing spaces
    column_name.erase(0, column_name.find_first_not_of(" \t"));
    column_name.erase(column_name.find_last_not_of(" \t") + 1);

    // Case-insensitive check for "CAST ("
    std::string lowered = column_name;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);

    if (lowered.find("cast (") == 0)
    {
        size_t start = column_name.find('(');
        if (start != std::string::npos && start + 1 < column_name.size())
        {
            // Get substring after '(' and trim it
            std::string inner = column_name.substr(start + 1);
            inner.erase(0, inner.find_first_not_of(" \t"));
            inner.erase(inner.find_last_not_of(" \t") + 1);

            // Split by space or AS if present
            size_t split_pos = inner.find_first_of(" )");
            if (split_pos != std::string::npos)
            {
                return inner.substr(0, split_pos);
            }
            return inner;
        }
    }

    return column_name;
}

bool *NestedLoopJoin::getSelectedRows(const TableResults &input_table, float value) const
{
    const size_t row_count = input_table.row_count;
    // bool *h_final_mask = new bool[row_count]();

    bool *d_combined_mask = nullptr;
    cudaMalloc(&d_combined_mask, row_count * sizeof(bool));
    cudaMemset(d_combined_mask, 0, row_count * sizeof(bool));

    int numThreads = 256;
    int numBlocks = (row_count + numThreads - 1) / numThreads;

    const size_t col_idx = input_table.getColumnIndex(column);
    const DataType col_type = input_table.columns[col_idx].type;
    uint8_t cond_code = 0;
    if (op == ">")
        cond_code = 1;
    else if (op == "<")
        cond_code = 2;
    else if (op == "=")
        cond_code = 3;
    else if (op == "!=")
        cond_code = 4;
    else if (op == "<=")
        cond_code = 5;
    else if (op == ">=")
        cond_code = 6;

    bool *d_temp_mask = nullptr;
    cudaMalloc(&d_temp_mask, row_count * sizeof(bool));
    cudaMemset(d_temp_mask, 0, row_count * sizeof(bool));

    if (col_type == DataType::FLOAT)
    {
        float *d_col_data = nullptr;
        cudaMalloc(&d_col_data, row_count * sizeof(float));
        cudaMemcpy(d_col_data, input_table.data[col_idx], row_count * sizeof(float), cudaMemcpyHostToDevice);
        filterKernel<float><<<numBlocks, numThreads>>>(d_col_data, d_temp_mask, row_count, value, cond_code);
        cudaError err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaFree(d_col_data);
    }

    bool *h_temp_mask = new bool[row_count];
    cudaMemcpy(h_temp_mask, d_temp_mask, row_count * sizeof(bool), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < row_count; i++)
    {
        if (h_temp_mask[i])
        {
            std::cout << i;
        }
    }

    cudaFree(d_temp_mask);
    return h_temp_mask;
}

TableResults NestedLoopJoin::applyNested(const TableResults &left_table, const TableResults &input_table) const
{
    if (input_table.row_count == 0)
    {
        return input_table;
    }
    float value = static_cast<float *>(left_table.data[0])[0];
    std::unique_ptr<bool[], void (*)(bool *)> h_selected_rows(
        getSelectedRows(input_table, value),
        [](bool *ptr)
        { delete[] ptr; });
    size_t selected_count = 0;
    const size_t row_count = input_table.row_count;
    for (size_t i = 0; i < row_count; i++)
    {
        if (h_selected_rows[i])
            selected_count++;
    }

    TableResults filtered_table;
    filtered_table.has_more = input_table.has_more;
    filtered_table.column_count = input_table.column_count;
    filtered_table.columns = input_table.columns;
    filtered_table.row_count = selected_count;
    filtered_table.batch_index = input_table.batch_index;
    filtered_table.data.resize(input_table.column_count);

    const size_t chunk_size = 10000;
    size_t num_chunks = (row_count + chunk_size - 1) / chunk_size;
    std::vector<cudaStream_t> streams(std::max<size_t>(1, num_chunks));
    for (auto &stream : streams)
    {
        cudaStreamCreate(&stream);
    }
    cudaStream_t prefix_stream;
    cudaStreamCreate(&prefix_stream);

    bool *d_mask = nullptr;
    cudaMalloc(&d_mask, row_count * sizeof(bool));
    cudaMemcpyAsync(d_mask, h_selected_rows.get(), row_count * sizeof(bool), cudaMemcpyHostToDevice, prefix_stream);

    std::vector<unsigned int> h_positions(row_count);
    unsigned int *d_positions = nullptr;
    cudaMalloc(&d_positions, row_count * sizeof(unsigned int));

    for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
    {
        size_t chunk_offset = chunk_idx * chunk_size;
        size_t chunk_rows = std::min(chunk_size, row_count - chunk_offset);
        int threads = 256;
        int blocks = (chunk_rows + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(unsigned int);
        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

        bool *d_chunk_mask = nullptr;
        cudaMalloc(&d_chunk_mask, chunk_rows * sizeof(bool));
        cudaMemcpyAsync(d_chunk_mask, h_selected_rows.get() + chunk_offset, chunk_rows * sizeof(bool),
                        cudaMemcpyHostToDevice, streams[stream_idx]);

        unsigned int *d_chunk_positions = nullptr;
        cudaMalloc(&d_chunk_positions, chunk_rows * sizeof(unsigned int));
        computeOutputPositions<<<blocks, threads, shared_mem, streams[stream_idx]>>>(d_chunk_mask, d_chunk_positions, chunk_rows);
        cudaMemcpyAsync(h_positions.data() + chunk_offset, d_chunk_positions, chunk_rows * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, streams[stream_idx]);

        cudaStreamSynchronize(streams[stream_idx]);
        cudaFree(d_chunk_mask);
        cudaFree(d_chunk_positions);
    }

    cudaMemcpyAsync(d_positions, h_positions.data(), row_count * sizeof(unsigned int), cudaMemcpyHostToDevice, prefix_stream);
    cudaStreamSynchronize(prefix_stream);

    for (size_t col_idx = 0; col_idx < input_table.column_count; col_idx++)
    {
        const DataType col_type = input_table.columns[col_idx].type;

        switch (col_type)
        {
        case DataType::FLOAT:
        {
            float *h_input_data = static_cast<float *>(input_table.data[col_idx]);
            std::unique_ptr<float[], void (*)(float *)> h_output_data(
                static_cast<float *>(malloc(selected_count * sizeof(float))),
                [](float *ptr)
                { free(ptr); });
            size_t output_offset = 0;

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
            {
                size_t chunk_offset = chunk_idx * chunk_size;
                size_t chunk_rows = std::min(chunk_size, row_count - chunk_offset);
                size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                float *d_input = nullptr;
                cudaMalloc(&d_input, chunk_rows * sizeof(float));
                cudaMemcpyAsync(d_input, h_input_data + chunk_offset, chunk_rows * sizeof(float),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                bool *d_chunk_mask = nullptr;
                cudaMalloc(&d_chunk_mask, chunk_rows * sizeof(bool));
                cudaMemcpyAsync(d_chunk_mask, h_selected_rows.get() + chunk_offset, chunk_rows * sizeof(bool),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                unsigned int *d_chunk_positions = nullptr;
                cudaMalloc(&d_chunk_positions, chunk_rows * sizeof(unsigned int));
                cudaMemcpyAsync(d_chunk_positions, h_positions.data() + chunk_offset, chunk_rows * sizeof(unsigned int),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                size_t chunk_selected_count = 0;
                for (size_t i = chunk_offset; i < chunk_offset + chunk_rows && i < row_count; i++)
                {
                    if (h_selected_rows[i])
                        chunk_selected_count++;
                }

                float *d_output = nullptr;
                if (chunk_selected_count > 0)
                {
                    cudaMalloc(&d_output, chunk_selected_count * sizeof(float));
                    int threads = 256;
                    int blocks = (chunk_rows + threads - 1) / threads;
                    copySelectedRowsKernel<float><<<blocks, threads, 0, streams[stream_idx]>>>(d_input, d_output, d_chunk_mask, d_chunk_positions, chunk_rows);
                    cudaMemcpyAsync(h_output_data.get() + output_offset, d_output, chunk_selected_count * sizeof(float),
                                    cudaMemcpyDeviceToHost, streams[stream_idx]);
                    output_offset += chunk_selected_count;
                }

                cudaStreamSynchronize(streams[stream_idx]);
                cudaFree(d_input);
                if (d_output)
                    cudaFree(d_output);
                cudaFree(d_chunk_mask);
                cudaFree(d_chunk_positions);
            }

            filtered_table.data[col_idx] = h_output_data.release();
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *h_input_data = static_cast<uint64_t *>(input_table.data[col_idx]);
            std::unique_ptr<uint64_t[], void (*)(uint64_t *)> h_output_data(
                static_cast<uint64_t *>(malloc(selected_count * sizeof(uint64_t))),
                [](uint64_t *ptr)
                { free(ptr); });
            size_t output_offset = 0;

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
            {
                size_t chunk_offset = chunk_idx * chunk_size;
                size_t chunk_rows = std::min(chunk_size, row_count - chunk_offset);
                size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                uint64_t *d_input = nullptr;
                cudaMalloc(&d_input, chunk_rows * sizeof(uint64_t));
                cudaMemcpyAsync(d_input, h_input_data + chunk_offset, chunk_rows * sizeof(uint64_t),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                bool *d_chunk_mask = nullptr;
                cudaMalloc(&d_chunk_mask, chunk_rows * sizeof(bool));
                cudaMemcpyAsync(d_chunk_mask, h_selected_rows.get() + chunk_offset, chunk_rows * sizeof(bool),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                unsigned int *d_chunk_positions = nullptr;
                cudaMalloc(&d_chunk_positions, chunk_rows * sizeof(unsigned int));
                cudaMemcpyAsync(d_chunk_positions, h_positions.data() + chunk_offset, chunk_rows * sizeof(unsigned int),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                size_t chunk_selected_count = 0;
                for (size_t i = chunk_offset; i < chunk_offset + chunk_rows && i < row_count; i++)
                {
                    if (h_selected_rows[i])
                        chunk_selected_count++;
                }

                uint64_t *d_output = nullptr;
                if (chunk_selected_count > 0)
                {
                    cudaMalloc(&d_output, chunk_selected_count * sizeof(uint64_t));
                    int threads = 256;
                    int blocks = (chunk_rows + threads - 1) / threads;
                    copySelectedRowsKernel<uint64_t><<<blocks, threads, 0, streams[stream_idx]>>>(d_input, d_output, d_chunk_mask, d_chunk_positions, chunk_rows);
                    cudaMemcpyAsync(h_output_data.get() + output_offset, d_output, chunk_selected_count * sizeof(uint64_t),
                                    cudaMemcpyDeviceToHost, streams[stream_idx]);
                    output_offset += chunk_selected_count;
                }

                cudaStreamSynchronize(streams[stream_idx]);
                cudaFree(d_input);
                if (d_output)
                    cudaFree(d_output);
                cudaFree(d_chunk_mask);
                cudaFree(d_chunk_positions);
            }

            filtered_table.data[col_idx] = h_output_data.release();
            break;
        }
        case DataType::STRING:
        {
            const char **h_input_strings = static_cast<const char **>(input_table.data[col_idx]);
            std::unique_ptr<const char *[], void (*)(const char **)> h_output_strings(
                static_cast<const char **>(malloc(selected_count * sizeof(char *))),
                [](const char **ptr)
                { free(ptr); });
            size_t output_offset = 0;

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++)
            {
                size_t chunk_offset = chunk_idx * chunk_size;
                size_t chunk_rows = std::min(chunk_size, row_count - chunk_offset);
                size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                const char **d_input_strings = nullptr;
                cudaMalloc(&d_input_strings, chunk_rows * sizeof(char *));
                cudaMemcpyAsync(d_input_strings, h_input_strings + chunk_offset, chunk_rows * sizeof(char *),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                bool *d_chunk_mask = nullptr;
                cudaMalloc(&d_chunk_mask, chunk_rows * sizeof(bool));
                cudaMemcpyAsync(d_chunk_mask, h_selected_rows.get() + chunk_offset, chunk_rows * sizeof(bool),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                unsigned int *d_chunk_positions = nullptr;
                cudaMalloc(&d_chunk_positions, chunk_rows * sizeof(unsigned int));
                cudaMemcpyAsync(d_chunk_positions, h_positions.data() + chunk_offset, chunk_rows * sizeof(unsigned int),
                                cudaMemcpyHostToDevice, streams[stream_idx]);

                size_t chunk_selected_count = 0;
                for (size_t i = chunk_offset; i < chunk_offset + chunk_rows && i < row_count; i++)
                {
                    if (h_selected_rows[i])
                        chunk_selected_count++;
                }

                const char **d_output_strings = nullptr;
                if (chunk_selected_count > 0)
                {
                    cudaMalloc(&d_output_strings, chunk_selected_count * sizeof(char *));
                    int threads = 256;
                    int blocks = (chunk_rows + threads - 1) / threads;
                    copySelectedStringRowsKernel<<<blocks, threads, 0, streams[stream_idx]>>>(d_input_strings, d_output_strings, d_chunk_mask, d_chunk_positions, chunk_rows);
                    cudaMemcpyAsync(h_output_strings.get() + output_offset, d_output_strings, chunk_selected_count * sizeof(char *),
                                    cudaMemcpyDeviceToHost, streams[stream_idx]);
                    output_offset += chunk_selected_count;
                }

                cudaStreamSynchronize(streams[stream_idx]);
                cudaFree(d_input_strings);
                if (d_output_strings)
                    cudaFree(d_output_strings);
                cudaFree(d_chunk_mask);
                cudaFree(d_chunk_positions);
            }

            filtered_table.data[col_idx] = h_output_strings.release();
            break;
        }
        default:
            for (auto &stream : streams)
                cudaStreamDestroy(stream);
            cudaStreamDestroy(prefix_stream);
            cudaFree(d_mask);
            cudaFree(d_positions);
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(col_type)));
        }
    }

    for (auto &stream : streams)
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    cudaStreamDestroy(prefix_stream);
    cudaFree(d_mask);
    cudaFree(d_positions);

    return filtered_table;
}

void NestedLoopJoin::print() const
{
    std::cout << "FILTER (";
    // for (size_t i = 0; i < conditions.size(); ++i)
    // {
    //     if (i > 0 && i - 1 < logical_ops.size())
    //     {
    //         std::cout << " " << logical_ops[i - 1] << " ";
    //     }
    //     std::cout << conditions[i].column << " " << conditions[i].op << " ";
    //     if (conditions[i].is_string)
    //     {
    //         std::cout << "'" << conditions[i].value << "'";
    //     }
    //     else
    //     {
    //         std::cout << conditions[i].value;
    //     }
    // }
    // std::cout << ")\n";
}