#include "physical_plan/order_by.hpp"

OrderBy::OrderBy(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("__order_by__");
    if (it != params.end())
    {
        // memory.main.table_1.id
        std::string text = it->second;
        if (text.substr(0, 12) == "memory.main.")
        {
            text = text.substr(12);
        }

        size_t pos = text.find('.');
        size_t pos2 = text.find(' ', pos);
        if (pos != std::string::npos && pos2 != std::string::npos)
        {
            table_name = text.substr(0, pos);
            col_name = text.substr(pos + 1, pos2 - pos - 1);
            std::string order = text.substr(pos2 + 1);
            if (order == "DESC")
            {
                is_Ascending = false;
            }
            else
            {
                is_Ascending = true;
            }
        }
    }
}
// Merge multiple sorted TableResults on GPU using multiple CUDA streams
TableResults OrderBy::mergeSortedBatchesOnGPU(const std::vector<TableResults> &batches)
{
    // Compute total rows and validate input
    size_t total_rows = 0;
    for (const auto &batch : batches)
    {
        total_rows += batch.row_count;
    }
    if (batches.empty() || total_rows == 0)
    {
        TableResults empty;
        empty.row_count = 0;
        empty.column_count = batches.empty() ? 0 : batches[0].column_count;
        empty.columns = batches.empty() ? std::vector<ColumnInfo>() : batches[0].columns;
        return empty;
    }

    // Create CUDA streams (one per batch + one for final merge)
    std::vector<cudaStream_t> streams(batches.size());
    cudaStream_t final_merge_stream;
    cudaStreamCreate(&final_merge_stream);
    for (size_t i = 0; i < batches.size(); ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate GPU memory for unified indices
    size_t *d_indices = nullptr;
    size_t *d_indicesTmp = nullptr;
    cudaMalloc(&d_indices, total_rows * sizeof(size_t));
    cudaMalloc(&d_indicesTmp, total_rows * sizeof(size_t));

    // Create batch index mapping: (batch_idx, row_idx)
    struct BatchIndex
    {
        size_t batch_idx;
        size_t row_idx;
    };
    std::vector<BatchIndex> h_batch_indices(total_rows);
    size_t offset = 0;
    for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
    {
        for (size_t row_idx = 0; row_idx < batches[batch_idx].row_count; ++row_idx)
        {
            h_batch_indices[offset] = {batch_idx, row_idx};
            offset++;
        }
    }
    BatchIndex *d_batch_indices = nullptr;
    cudaMalloc(&d_batch_indices, total_rows * sizeof(BatchIndex));
    cudaMemcpyAsync(d_batch_indices, h_batch_indices.data(), total_rows * sizeof(BatchIndex), cudaMemcpyHostToDevice, final_merge_stream);

    // Find key column index
    size_t col_idx = batches[0].getColumnIndex(col_name);
    switch (batches[0].columns[col_idx].type)
    {
    case DataType::FLOAT:
    {
        float *d_data = nullptr;
        cudaMalloc(&d_data, total_rows * sizeof(float));
        // Copy keys for each batch in its own stream
        offset = 0;
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            float *batch_data = static_cast<float *>(batches[batch_idx].data[col_idx]);
            cudaMemcpyAsync(d_data + offset, batch_data, batches[batch_idx].row_count * sizeof(float),
                            cudaMemcpyHostToDevice, streams[batch_idx]);
            offset += batches[batch_idx].row_count;
        }
        // Synchronize batch streams before merging
        for (auto &stream : streams)
        {
            cudaStreamSynchronize(stream);
        }
        // Merge in final stream
        int threadsPerBlock = 256;
        for (int width = 1; width < total_rows; width *= 2)
        {
            int blocks = (total_rows + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;
            mergeSortKernel<float><<<gridSize, threadsPerBlock, 0, final_merge_stream>>>(
                d_data, d_indices, d_indicesTmp, total_rows, width, is_Ascending);
        }
        cudaFree(d_data);
        break;
    }
    case DataType::DATETIME:
    {
        uint64_t *d_data = nullptr;
        cudaMalloc(&d_data, total_rows * sizeof(uint64_t));
        offset = 0;
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            uint64_t *batch_data = static_cast<uint64_t *>(batches[batch_idx].data[col_idx]);
            cudaMemcpyAsync(d_data + offset, batch_data, batches[batch_idx].row_count * sizeof(uint64_t),
                            cudaMemcpyHostToDevice, streams[batch_idx]);
            offset += batches[batch_idx].row_count;
        }
        for (auto &stream : streams)
        {
            cudaStreamSynchronize(stream);
        }
        int threadsPerBlock = 256;
        for (int width = 1; width < total_rows; width *= 2)
        {
            int blocks = (total_rows + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;
            mergeSortKernel<uint64_t><<<gridSize, threadsPerBlock, 0, final_merge_stream>>>(
                d_data, d_indices, d_indicesTmp, total_rows, width, is_Ascending);
        }
        cudaFree(d_data);
        break;
    }
    case DataType::STRING:
    {
        const char **d_data = nullptr;
        cudaMalloc(&d_data, total_rows * sizeof(char *));
        char **d_strings = new char *[total_rows];
        offset = 0;
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            const char **batch_strings = static_cast<const char **>(batches[batch_idx].data[col_idx]);
            for (size_t row_idx = 0; row_idx < batches[batch_idx].row_count; ++row_idx)
            {
                size_t len = strlen(batch_strings[row_idx]) + 1;
                cudaMalloc(&d_strings[offset], len);
                cudaMemcpyAsync(d_strings[offset], batch_strings[row_idx], len, cudaMemcpyHostToDevice, streams[batch_idx]);
                cudaMemcpyAsync(&d_data[offset], &d_strings[offset], sizeof(char *), cudaMemcpyHostToDevice, streams[batch_idx]);
                offset++;
            }
        }
        for (auto &stream : streams)
        {
            cudaStreamSynchronize(stream);
        }
        int threadsPerBlock = 256;
        for (int width = 1; width < total_rows; width *= 2)
        {
            int blocks = (total_rows + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;
            mergeSortKernel<char *><<<gridSize, threadsPerBlock, 0, final_merge_stream>>>(
                const_cast<char **>(d_data), d_indices, d_indicesTmp, total_rows, width, is_Ascending);
        }
        for (size_t i = 0; i < total_rows; ++i)
        {
            cudaFree(d_strings[i]);
        }
        delete[] d_strings;
        cudaFree(d_data);
        break;
    }
    default:
        for (auto &stream : streams)
        {
            cudaStreamDestroy(stream);
        }
        cudaStreamDestroy(final_merge_stream);
        cudaFree(d_batch_indices);
        cudaFree(d_indices);
        cudaFree(d_indicesTmp);
        throw std::runtime_error("Unsupported data type");
    }

    // Copy merged batch indices back to host
    std::vector<BatchIndex> h_merged_batch_indices(total_rows);
    cudaMemcpyAsync(h_merged_batch_indices.data(), d_batch_indices, total_rows * sizeof(BatchIndex),
                    cudaMemcpyDeviceToHost, final_merge_stream);

    // Synchronize final stream
    cudaStreamSynchronize(final_merge_stream);

    // Clean up GPU memory and streams
    cudaFree(d_batch_indices);
    cudaFree(d_indices);
    cudaFree(d_indicesTmp);
    for (auto &stream : streams)
    {
        cudaStreamDestroy(stream);
    }
    cudaStreamDestroy(final_merge_stream);

    // Construct output TableResults
    TableResults result;
    result.column_count = batches[0].column_count;
    result.has_more = batches[0].has_more;
    result.row_count = total_rows;
    result.columns = batches[0].columns;
    result.data.resize(result.column_count);

    // Allocate host memory for output
    for (size_t col = 0; col < result.column_count; ++col)
    {
        switch (result.columns[col].type)
        {
        case DataType::FLOAT:
        {
            float *h_output = static_cast<float *>(malloc(total_rows * sizeof(float)));
            offset = 0;
            for (const auto &idx : h_merged_batch_indices)
            {
                float *batch_data = static_cast<float *>(batches[idx.batch_idx].data[col]);
                h_output[offset++] = batch_data[idx.row_idx];
            }
            result.data[col] = h_output;
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *h_output = static_cast<uint64_t *>(malloc(total_rows * sizeof(uint64_t)));
            offset = 0;
            for (const auto &idx : h_merged_batch_indices)
            {
                uint64_t *batch_data = static_cast<uint64_t *>(batches[idx.batch_idx].data[col]);
                h_output[offset++] = batch_data[idx.row_idx];
            }
            result.data[col] = h_output;
            break;
        }
        case DataType::STRING:
        {
            const char **h_output = static_cast<const char **>(malloc(total_rows * sizeof(char *)));
            offset = 0;
            for (const auto &idx : h_merged_batch_indices)
            {
                const char **batch_data = static_cast<const char **>(batches[idx.batch_idx].data[col]);
                h_output[offset++] = batch_data[idx.row_idx];
            }
            result.data[col] = h_output;
            break;
        }
        default:
            break;
        }
    }

    return result;
}

std::vector<size_t> OrderBy::getSortedIndex(const TableResults &input_table)
{
    size_t n = input_table.row_count;

    size_t *d_indices = nullptr;
    size_t *d_indicesTmp = nullptr;

    cudaMalloc(&d_indices, n * sizeof(size_t));
    cudaMalloc(&d_indicesTmp, n * sizeof(size_t));

    size_t *h_indices = new size_t[n];
    for (int i = 0; i < n; i++)
    {
        h_indices[i] = i;
    }

    cudaMemcpy(d_indices, h_indices, n * sizeof(size_t), cudaMemcpyHostToDevice);
    delete[] h_indices;

    int threadsPerBlock = 256;

    size_t index = input_table.getColumnIndex(col_name);
    switch (input_table.columns[index].type)
    {
    case DataType::FLOAT:
    {
        float *d_data = nullptr;
        cudaMalloc(&d_data, n * sizeof(float));
        cudaMemcpy(d_data, input_table.data[index], n * sizeof(float), cudaMemcpyHostToDevice);

        for (int width = 1; width < n; width *= 2)
        {
            int blocks = (n + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;

            mergeSortKernel<float><<<gridSize, threadsPerBlock>>>(
                d_data, d_indices, d_indicesTmp, n, width, is_Ascending);

            cudaDeviceSynchronize();
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaFree(d_data);
        break;
    }
    case DataType::DATETIME:
    {
        uint64_t *d_data = nullptr;
        cudaMalloc(&d_data, n * sizeof(uint64_t));
        cudaMemcpy(d_data, input_table.data[index], n * sizeof(uint64_t), cudaMemcpyHostToDevice);

        for (int width = 1; width < n; width *= 2)
        {
            int blocks = (n + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;

            mergeSortKernel<uint64_t><<<gridSize, threadsPerBlock>>>(
                d_data, d_indices, d_indicesTmp, n, width, is_Ascending);

            cudaDeviceSynchronize();
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaFree(d_data);
        break;
    }
    case DataType::STRING:
    {
        const char **d_data = nullptr;
        cudaMalloc(&d_data, n * sizeof(char *));

        const char **host_strings = static_cast<const char **>(input_table.data[index]);

        char **d_strings = new char *[input_table.row_count];

        for (size_t i = 0; i < input_table.row_count; i++)
        {
            size_t len = strlen(host_strings[i]) + 1;
            cudaMalloc(&d_strings[i], len);
            cudaMemcpy(d_strings[i], host_strings[i], len, cudaMemcpyHostToDevice);
            cudaMemcpy(&d_data[i], &d_strings[i], sizeof(char *), cudaMemcpyHostToDevice);
        }

        for (int width = 1; width < n; width *= 2)
        {
            int blocks = (n + width * 2 - 1) / (width * 2);
            int gridSize = (blocks + threadsPerBlock - 1) / threadsPerBlock;

            mergeSortKernel<char *><<<gridSize, threadsPerBlock>>>(
                const_cast<char **>(d_data), d_indices, d_indicesTmp, n, width, is_Ascending);

            cudaDeviceSynchronize();
        }
        for (size_t i = 0; i < n; i++)
        {
            cudaFree(d_strings[i]);
        }

        delete[] d_strings;
        cudaFree(d_data);
    }
    default:
        break;
    }

    std::vector<size_t> sorted_indices(n);
    cudaMemcpy(sorted_indices.data(), d_indices, n * sizeof(size_t), cudaMemcpyDeviceToHost);

    cudaFree(d_indices);
    cudaFree(d_indicesTmp);

    return sorted_indices;
}

TableResults OrderBy::executeOrderBy(const TableResults &input_table)
{
    std::vector<size_t> sorted_indices = getSortedIndex(input_table);

    TableResults result;
    result.column_count = input_table.columns.size();
    result.has_more = input_table.has_more;
    result.row_count = sorted_indices.size();
    result.columns = input_table.columns;
    result.data.resize(input_table.columns.size());

    size_t *d_idx = nullptr;
    cudaMalloc(&d_idx, result.row_count * sizeof(size_t));
    cudaMemcpy(d_idx, sorted_indices.data(), result.row_count * sizeof(size_t), cudaMemcpyHostToDevice);

    int numThreads = 256;
    int numBlocks = (result.row_count + numThreads - 1) / numThreads;

    for (size_t i = 0; i < input_table.columns.size(); ++i)
    {
        switch (input_table.columns[i].type)
        {
        case DataType::FLOAT:
        {
            float *d_input, *d_output;
            cudaMalloc(&d_input, input_table.row_count * sizeof(float));
            cudaMalloc(&d_output, result.row_count * sizeof(float));
            cudaMemcpy(d_input, input_table.data[i], input_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
            getRowsKernel<float><<<numBlocks, numThreads>>>(
                d_input, d_idx, d_output, result.row_count);
            cudaDeviceSynchronize();
            float *h_output_data = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(float), cudaMemcpyDeviceToHost);
            result.data[i] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *d_input, *d_output;
            cudaMalloc(&d_input, input_table.row_count * sizeof(uint64_t));
            cudaMalloc(&d_output, result.row_count * sizeof(uint64_t));
            cudaMemcpy(d_input, input_table.data[i], input_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
            getRowsKernel<uint64_t><<<numBlocks, numThreads>>>(
                d_input, d_idx, d_output, result.row_count);
            cudaDeviceSynchronize();
            uint64_t *h_output_data = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            result.data[i] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::STRING:
        {
            const char **h_input_strings = static_cast<const char **>(input_table.data[i]);
            const char **h_output_strings = static_cast<const char **>(malloc(result.row_count * sizeof(char *)));

            const char **d_input_strings;
            cudaMalloc(&d_input_strings, input_table.row_count * sizeof(char *));
            cudaMemcpy(d_input_strings, h_input_strings, input_table.row_count * sizeof(char *), cudaMemcpyHostToDevice);

            const char **d_output_strings;
            cudaMalloc(&d_output_strings, result.row_count * sizeof(char *));
            getRowsKernel<const char *><<<numBlocks, numThreads>>>(
                d_input_strings, d_idx, d_output_strings, result.row_count);

            cudaDeviceSynchronize();

            cudaMemcpy(h_output_strings, d_output_strings, result.row_count * sizeof(char *), cudaMemcpyDeviceToHost);

            result.data[i] = h_output_strings;
            cudaFree(d_input_strings);
            cudaFree(d_output_strings);
            break;
        }
        default:
            break;
        }
    }

    cudaFree(d_idx);
    return result;
}

void OrderBy::print() const
{
    // std::cout << "OrderBy: " << col_name << " " << order;
}

void OrderBy::write_intermideate(const std::vector<TableResults> &batches)
{
    std::filesystem::create_directories("temp");

    for (size_t i = 0; i < batches.size(); i++)
    {
        const TableResults &batch = batches[i];
        std::string filename = "temp/order_by_batch_" + std::to_string(i) + ".bin";

        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile)
        {
            throw std::runtime_error("Failed to open temporary file: " + filename);
        }

        // Write batch metadata in a compact format
        uint32_t row_count = static_cast<uint32_t>(batch.row_count);
        uint16_t column_count = static_cast<uint16_t>(batch.column_count);
        outfile.write(reinterpret_cast<const char *>(&row_count), sizeof(uint32_t));
        outfile.write(reinterpret_cast<const char *>(&column_count), sizeof(uint16_t));

        // Write column information in a compact format
        for (const auto &col : batch.columns)
        {
            uint8_t name_len = static_cast<uint8_t>(col.name.length());
            outfile.write(reinterpret_cast<const char *>(&name_len), sizeof(uint8_t));
            outfile.write(col.name.c_str(), name_len);
            uint8_t type = static_cast<uint8_t>(col.type);
            outfile.write(reinterpret_cast<const char *>(&type), sizeof(uint8_t));
        }

        // Write data for each column in a compact format
        for (size_t col_idx = 0; col_idx < batch.column_count; col_idx++)
        {
            switch (batch.columns[col_idx].type)
            {
            case DataType::FLOAT:
            {
                float *data = static_cast<float *>(batch.data[col_idx]);
                outfile.write(reinterpret_cast<const char *>(data), batch.row_count * sizeof(float));
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t *data = static_cast<uint64_t *>(batch.data[col_idx]);
                outfile.write(reinterpret_cast<const char *>(data), batch.row_count * sizeof(uint64_t));
                break;
            }
            case DataType::STRING:
            {
                const char **data = static_cast<const char **>(batch.data[col_idx]);
                for (size_t row = 0; row < batch.row_count; row++)
                {
                    uint16_t str_len = static_cast<uint16_t>(strlen(data[row]));
                    outfile.write(reinterpret_cast<const char *>(&str_len), sizeof(uint16_t));
                    outfile.write(data[row], str_len);
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type in write_intermediate");
            }
        }

        outfile.close();
    }
}

TableResults OrderBy::read_intermediate(const std::string &filename)
{
    std::ifstream infile(filename, std::ios::binary);
    if (!infile)
    {
        throw std::runtime_error("Failed to open temporary file: " + filename);
    }

    TableResults result;

    // Read batch metadata
    uint32_t row_count;
    uint16_t column_count;
    infile.read(reinterpret_cast<char *>(&row_count), sizeof(uint32_t));
    infile.read(reinterpret_cast<char *>(&column_count), sizeof(uint16_t));

    result.row_count = row_count;
    result.column_count = column_count;
    result.columns.resize(column_count);
    result.data.resize(column_count);

    // Read column information
    for (size_t i = 0; i < column_count; i++)
    {
        uint8_t name_len;
        infile.read(reinterpret_cast<char *>(&name_len), sizeof(uint8_t));
        std::string col_name(name_len, '\0');
        infile.read(&col_name[0], name_len);

        uint8_t type;
        infile.read(reinterpret_cast<char *>(&type), sizeof(uint8_t));

        result.columns[i].name = col_name;
        result.columns[i].type = static_cast<DataType>(type);
    }

    // Read data for each column
    for (size_t col_idx = 0; col_idx < column_count; col_idx++)
    {
        switch (result.columns[col_idx].type)
        {
        case DataType::FLOAT:
        {
            float *data = static_cast<float *>(malloc(row_count * sizeof(float)));
            infile.read(reinterpret_cast<char *>(data), row_count * sizeof(float));
            result.data[col_idx] = data;
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *data = static_cast<uint64_t *>(malloc(row_count * sizeof(uint64_t)));
            infile.read(reinterpret_cast<char *>(data), row_count * sizeof(uint64_t));
            result.data[col_idx] = data;
            break;
        }
        case DataType::STRING:
        {
            const char **data = static_cast<const char **>(malloc(row_count * sizeof(char *)));
            for (size_t row = 0; row < row_count; row++)
            {
                uint16_t str_len;
                infile.read(reinterpret_cast<char *>(&str_len), sizeof(uint16_t));
                char *str = static_cast<char *>(malloc(str_len + 1));
                infile.read(str, str_len);
                str[str_len] = '\0';
                data[row] = str;
            }
            result.data[col_idx] = data;
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in read_intermediate");
        }
    }

    infile.close();
    return result;
}

TableResults OrderBy::merge_sorted_files()
{
    std::vector<TableResults> batches;
    size_t batch_count = 0;

    // Read all batch files
    while (true)
    {
        std::string filename = "temp/order_by_batch_" + std::to_string(batch_count) + ".bin";
        if (!std::filesystem::exists(filename))
        {
            break;
        }
        batches.push_back(read_intermediate(filename));
        batch_count++;
    }

    if (batches.empty())
    {
        TableResults empty;
        empty.row_count = 0;
        empty.column_count = 0;
        return empty;
    }

    // Merge all batches using the existing GPU merge function
    TableResults result = mergeSortedBatchesOnGPU(batches);

    // Clean up temporary files
    for (size_t i = 0; i < batch_count; i++)
    {
        std::string filename = "temp/order_by_batch_" + std::to_string(i) + ".bin";
        std::filesystem::remove(filename);
    }

    return result;
}