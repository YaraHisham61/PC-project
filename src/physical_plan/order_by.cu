#include "physical_plan/order_by.hpp"

OrderBy::OrderBy(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("__order_by__");
    if (it != params.end())
    {
        // memory.main.table_1.id
        std::string text = it->second;
        text = text.substr(12);
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
        std::cout << "table_name: " << table_name << std::endl;
        std::cout << "col_name: " << col_name << std::endl;
        std::cout << "order: " << is_Ascending << std::endl;
    }
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

    // Clean up
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
                d_input,
                d_idx,
                d_output,
                result.row_count);
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
                d_input,
                d_idx,
                d_output,
                result.row_count);
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
                d_input_strings,
                d_idx,
                d_output_strings,
                result.row_count);

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