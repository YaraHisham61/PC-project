#include "physical_plan/join.hpp"

HashJoin::HashJoin(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("Conditions");
    if (it != params.end())
    {
        size_t pos = it->second.find('=');
        if (pos != std::string::npos)
        {
            col_table_left = it->second.substr(0, pos - 1);
            col_table_right = it->second.substr(pos + 2);
        }
        std::cout << "col_table_left: " << col_table_left << std::endl;
        std::cout << "col_table_right: " << col_table_right << std::endl;
    }
}

void HashJoin::getIndexOfSelectedRows(const TableResults &left_table, const TableResults &right_table,
                                      std::vector<size_t> &left_indices, std::vector<size_t> &right_indices)
{

    size_t index_left = left_table.getColumnIndex(col_table_left);
    size_t index_right = right_table.getColumnIndex(col_table_right);
    ColumnInfo left_col = left_table.columns[index_left];
    ColumnInfo right_col = right_table.columns[index_right];

    if (left_col.type != right_col.type)
    {
        throw std::runtime_error("Join columns have different types");
    }

    int numThreads = 256;
    int numBlocks = (left_table.row_count + numThreads - 1) / numThreads;
    size_t shared_mem_size = ((numThreads + 31) / 32) * getDataTypeNumBytes(left_col.type);

    size_t row_count_left = left_table.row_count,
           row_count_right = right_table.row_count;
    size_t max_pairs = row_count_left * row_count_right;

    size_t *d_left_idx = nullptr;
    size_t *d_right_idx = nullptr;
    unsigned long long *d_count = nullptr;

    cudaMalloc(&d_left_idx, max_pairs * sizeof(size_t));
    cudaMalloc(&d_right_idx, max_pairs * sizeof(size_t));
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    if (left_col.type == DataType::FLOAT)
    {
        float *d_left_data = nullptr;
        float *d_right_data = nullptr;
        cudaMalloc(&d_left_data, row_count_left * sizeof(float));
        cudaMalloc(&d_right_data, row_count_right * sizeof(float));
        cudaMemcpy(d_left_data, left_table.data[index_left], row_count_left * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right_data, right_table.data[index_right], row_count_right * sizeof(float), cudaMemcpyHostToDevice);
        size_t shared_mem_size2 = numThreads * sizeof(float);

        hashJoinKernel<float><<<numBlocks, numThreads, shared_mem_size2>>>(
            d_left_data, d_right_data,
            row_count_left, row_count_right,
            d_left_idx, d_right_idx, d_count);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaFree(d_left_data);
        cudaFree(d_right_data);
    }
    else if (left_col.type == DataType::DATETIME)
    {
        uint64_t *d_left_data = nullptr;
        uint64_t *d_right_data = nullptr;
        cudaMalloc(&d_left_data, left_table.row_count * sizeof(uint64_t));
        cudaMalloc(&d_right_data, right_table.row_count * sizeof(uint64_t));

        cudaMemcpy(d_left_data, left_table.data[index_left], left_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right_data, right_table.data[index_right], right_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

        hashJoinKernel<uint64_t><<<numBlocks, numThreads, shared_mem_size>>>(
            d_left_data, d_right_data,
            row_count_left, row_count_right,
            d_left_idx, d_right_idx, d_count);
        cudaDeviceSynchronize();
        cudaFree(d_left_data);
        cudaFree(d_right_data);
    }
    else if (left_col.type == DataType::STRING)
    {
        const char **d_left_data = nullptr;
        const char **d_right_data = nullptr;
        cudaMalloc(&d_left_data, left_table.row_count * sizeof(char *));
        cudaMalloc(&d_right_data, right_table.row_count * sizeof(char *));

        const char **host_strings_left = static_cast<const char **>(left_table.data[index_left]);
        const char **host_strings_right = static_cast<const char **>(right_table.data[index_right]);

        char **d_strings_left = new char *[left_table.row_count];
        char **d_strings_right = new char *[right_table.row_count];

        for (size_t i = 0; i < left_table.row_count; i++)
        {
            size_t len = strlen(host_strings_left[i]) + 1;
            cudaMalloc(&d_strings_left[i], len);
            cudaMemcpy(d_strings_left[i], host_strings_left[i], len, cudaMemcpyHostToDevice);
            cudaMemcpy(&d_left_data[i], &d_strings_left[i], sizeof(char *), cudaMemcpyHostToDevice);
        }

        for (size_t i = 0; i < right_table.row_count; i++)
        {
            size_t len = strlen(host_strings_right[i]) + 1;
            cudaMalloc(&d_strings_right[i], len);
            cudaMemcpy(d_strings_right[i], host_strings_right[i], len, cudaMemcpyHostToDevice);
            cudaMemcpy(&d_right_data[i], &d_strings_right[i], sizeof(char *), cudaMemcpyHostToDevice);
        }
        hashJoinKernel<const char *><<<numBlocks, numThreads, shared_mem_size>>>(
            d_left_data, d_right_data,
            row_count_left, row_count_right,
            d_left_idx, d_right_idx, d_count);
        cudaDeviceSynchronize();
        for (size_t i = 0; i < left_table.row_count; i++)
        {
            cudaFree(d_strings_left[i]);
        }
        for (size_t i = 0; i < right_table.row_count; i++)
        {
            cudaFree(d_strings_right[i]);
        }
        delete[] d_strings_left;
        delete[] d_strings_right;
        cudaFree(d_left_data);
        cudaFree(d_right_data);
    }
    else
    {
        throw std::runtime_error("Unsupported join column type");
    }
    cudaDeviceSynchronize();
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    left_indices.resize(h_count); 
    right_indices.resize(h_count);

    cudaMemcpy(left_indices.data(), d_left_idx, h_count * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_left_idx);

    cudaMemcpy(right_indices.data(), d_right_idx, h_count * sizeof(size_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_right_idx);
    cudaFree(d_count);
}
TableResults HashJoin::executeJoin(const TableResults &left_table, const TableResults &right_table)
{
    std::vector<size_t> left_indices(left_table.row_count);
    std::vector<size_t> right_indices(right_table.row_count);
    getIndexOfSelectedRows(left_table, right_table, left_indices, right_indices);

    TableResults result;
    result.column_count = left_table.columns.size() + right_table.columns.size();
    result.row_count = left_indices.size();
    result.columns = left_table.columns;
    result.columns.insert(result.columns.end(), right_table.columns.begin(), right_table.columns.end());
    result.data.resize(left_table.columns.size() + right_table.columns.size());

    for (size_t i = 0; i < result.columns.size(); ++i)
    {
        result.columns[i].idx = i;
    }

    size_t *d_left_idx = nullptr;
    size_t *d_right_idx = nullptr;
    cudaMalloc(&d_left_idx, result.row_count * sizeof(size_t));
    cudaMalloc(&d_right_idx, result.row_count * sizeof(size_t));
    cudaMemcpy(d_left_idx, left_indices.data(), result.row_count * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_idx, right_indices.data(), result.row_count * sizeof(size_t), cudaMemcpyHostToDevice);

    int numThreads = 256;
    int numBlocks = (result.row_count + numThreads - 1) / numThreads;

    size_t col_idx = 0;
    for (size_t i = 0; i < left_table.columns.size(); ++i)
    {

        switch (left_table.columns[i].type)
        {
        case DataType::FLOAT:
        {
            float *d_input, *d_output;
            cudaMalloc(&d_input, left_table.row_count * sizeof(float));
            cudaMalloc(&d_output, result.row_count * sizeof(float));
            cudaMemcpy(d_input, left_table.data[i], left_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
            getRowsKernel<float><<<numBlocks, numThreads>>>(
                d_input,
                d_left_idx,
                d_output,
                result.row_count);
            cudaDeviceSynchronize();
            float *h_output_data = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(float), cudaMemcpyDeviceToHost);
            // for (size_t j = 0; j < result.row_count; ++j)
            // {
            //     std::cout << "h_output_data[" << j << "] = " << h_output_data[j] << std::endl;
            // }
            result.data[col_idx] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *d_input, *d_output;
            cudaMalloc(&d_input, left_table.row_count * sizeof(uint64_t));
            cudaMalloc(&d_output, result.row_count * sizeof(uint64_t));
            cudaMemcpy(d_input, left_table.data[i], left_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
            getRowsKernel<uint64_t><<<numBlocks, numThreads>>>(
                d_input,
                d_left_idx,
                d_output,
                result.row_count);
            cudaDeviceSynchronize();
            uint64_t *h_output_data = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            result.data[col_idx] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::STRING:
        {
            const char **h_input_strings = static_cast<const char **>(left_table.data[i]);
            const char **h_output_strings = static_cast<const char **>(malloc(result.row_count * sizeof(char *)));

            const char **d_input_strings;
            cudaMalloc(&d_input_strings, left_table.row_count * sizeof(char *));
            cudaMemcpy(d_input_strings, h_input_strings, left_table.row_count * sizeof(char *), cudaMemcpyHostToDevice);

            const char **d_output_strings;
            cudaMalloc(&d_output_strings, result.row_count * sizeof(char *));
            getRowsKernel<const char *><<<numBlocks, numThreads>>>(
                d_input_strings,
                d_left_idx,
                d_output_strings,
                result.row_count);

            cudaDeviceSynchronize();

            cudaMemcpy(h_output_strings, d_output_strings, result.row_count * sizeof(char *), cudaMemcpyDeviceToHost);

            result.data[col_idx] = h_output_strings;
            cudaFree(d_input_strings);
            cudaFree(d_output_strings);
            break;
        }
        default:
            break;
        }
        col_idx++;
    }
    for (size_t i = 0; i < right_table.columns.size(); ++i)
    {
        switch (right_table.columns[i].type)
        {
        case DataType::FLOAT:
        {
            float *d_input, *d_output;
            cudaMalloc(&d_input, right_table.row_count * sizeof(float));
            cudaMalloc(&d_output, result.row_count * sizeof(float));
            cudaMemcpy(d_input, right_table.data[i], right_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
            getRowsKernel<float><<<numBlocks, numThreads>>>(
                d_input,
                d_right_idx,
                d_output,
                result.row_count);
            cudaDeviceSynchronize();
            float *h_output_data = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(float), cudaMemcpyDeviceToHost);
            result.data[col_idx] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *d_input, *d_output;
            cudaMalloc(&d_input, right_table.row_count * sizeof(uint64_t));
            cudaMalloc(&d_output, result.row_count * sizeof(uint64_t));
            cudaMemcpy(d_input, right_table.data[i], right_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
            getRowsKernel<uint64_t><<<numBlocks, numThreads>>>(
                d_input,
                d_right_idx,
                d_output,
                result.row_count);

            cudaDeviceSynchronize();
            uint64_t *h_output_data = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
            cudaMemcpy(h_output_data, d_output, result.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            result.data[col_idx] = h_output_data;
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::STRING:
        {
            const char **h_input_strings = static_cast<const char **>(right_table.data[i]);
            const char **h_output_strings = static_cast<const char **>(malloc(result.row_count * sizeof(char *)));

            const char **d_input_strings;
            cudaMalloc(&d_input_strings, right_table.row_count * sizeof(char *));
            cudaMemcpy(d_input_strings, h_input_strings, right_table.row_count * sizeof(char *), cudaMemcpyHostToDevice);

            const char **d_output_strings;
            cudaMalloc(&d_output_strings, result.row_count * sizeof(char *));
            getRowsKernel<const char *><<<numBlocks, numThreads>>>(
                d_input_strings,
                d_right_idx,
                d_output_strings,
                result.row_count);

            cudaDeviceSynchronize();

            cudaMemcpy(h_output_strings, d_output_strings, result.row_count * sizeof(char *), cudaMemcpyDeviceToHost);

            result.data[col_idx] = h_output_strings;
            cudaFree(d_input_strings);
            cudaFree(d_output_strings);
            break;
        }
        default:
            break;
        }
        col_idx++;
    }
    cudaFree(d_left_idx);
    cudaFree(d_right_idx);
    return result;
}

void HashJoin::print() const
{
    std::cout << "HashJoin: " << col_table_left << " = " << col_table_right << std::endl;
    // for (const auto &child : children)
    // {
    //     child->print(os, indent + 2);
    // }
}
