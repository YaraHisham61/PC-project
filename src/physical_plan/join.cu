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
TableResults HashJoin::executeJoin(const TableResults &left_table, const TableResults &right_table)
{
    size_t index_left = left_table.getColumnIndex(col_table_left);
    size_t index_right = right_table.getColumnIndex(col_table_right);
    ColumnInfo left_col = left_table.columns[index_left];
    ColumnInfo right_col = right_table.columns[index_left];

    if (left_col.type != right_col.type)
    {
        throw std::runtime_error("Join columns have different types");
    }

    int numThreads = 256;
    int numBlocks = (left_table.row_count + numThreads - 1) / numThreads;
    size_t shared_mem_size = ((numThreads + 31) / 32) * getDataTypeNumBytes(left_col.type);

    bool *h_output_mask_left = new bool[left_table.row_count]();
    bool *h_output_mask_right = new bool[right_table.row_count]();
    bool *d_output_mask_left = nullptr;
    bool *d_output_mask_right = nullptr;
    cudaMalloc(&d_output_mask_left, left_table.row_count * sizeof(bool));
    cudaMalloc(&d_output_mask_right, right_table.row_count * sizeof(bool));
    cudaMemset(d_output_mask_left, 0, left_table.row_count * sizeof(bool));
    cudaMemset(d_output_mask_right, 0, right_table.row_count * sizeof(bool));
    if (left_col.type == DataType::FLOAT)
    {
        float *d_left_data = nullptr;
        float *d_right_data = nullptr;
        cudaMalloc(&d_left_data, left_table.row_count * sizeof(float));
        cudaMalloc(&d_right_data, right_table.row_count * sizeof(float));
        cudaMemcpy(d_left_data, left_table.data[index_left], left_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_right_data, right_table.data[index_right], right_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
        size_t shared_mem_size2 = numThreads * sizeof(float);

        hashJoinKernel<float><<<numBlocks, numThreads, shared_mem_size2>>>(
            d_left_data, d_right_data,
            d_output_mask_left, d_output_mask_right,
            left_table.row_count, right_table.row_count);
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
            d_output_mask_left, d_output_mask_right,
            left_table.row_count, right_table.row_count);
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
            d_output_mask_left, d_output_mask_right,
            left_table.row_count, right_table.row_count);
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
    cudaMemcpy(h_output_mask_left, d_output_mask_left, left_table.row_count * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_mask_right, d_output_mask_right, right_table.row_count * sizeof(bool), cudaMemcpyDeviceToHost);
    int left = 0, right = 0;
    for (size_t i = 0; i < left_table.row_count; i++)
    {
        if (h_output_mask_left[i])
        {
            left++;
        }
    }
    for (size_t i = 0; i < right_table.row_count; i++)
    {
        if (h_output_mask_right[i])
        {
            right++;
        }
    }
    std::cout << "Left matches: " << left << std::endl;
    std::cout << "Right matches: " << right << std::endl;
    cudaFree(d_output_mask_left);
    cudaFree(d_output_mask_right);
    return TableResults();
}

void HashJoin::print() const
{
    std::cout << "HashJoin: " << col_table_left << " = " << col_table_right << std::endl;
    // for (const auto &child : children)
    // {
    //     child->print(os, indent + 2);
    // }
}
