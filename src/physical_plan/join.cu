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

    const size_t chunk_size = 10000;
    size_t row_count_left = left_table.row_count;
    size_t row_count_right = right_table.row_count;
    size_t num_chunks = (row_count_right + chunk_size - 1) / chunk_size;
    const size_t max_streams = 16;
    size_t num_streams = std::min(num_chunks, max_streams);
    std::vector<cudaStream_t> streams(num_streams);
    std::vector<void *> allocations;

    for (auto &stream : streams)
    {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
        }
    }

    int numThreads = 256;
    size_t shared_mem_size = ((numThreads + 31) / 32) * getDataTypeNumBytes(left_col.type);

    try
    {
        if (left_col.type == DataType::FLOAT)
        {
            float *d_left_data = nullptr;
            cudaError_t err = cudaMalloc(&d_left_data, row_count_left * sizeof(float));
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to allocate d_left_data (FLOAT): ") + cudaGetErrorString(err));
            }
            allocations.push_back(d_left_data);

            err = cudaMemcpy(d_left_data, left_table.data[index_left], row_count_left * sizeof(float), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to copy left_table.data to d_left_data (FLOAT): ") + cudaGetErrorString(err));
            }

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                size_t right_start = chunk_idx * chunk_size;
                size_t right_rows = std::min(chunk_size, row_count_right - right_start);
                if (right_rows == 0)
                    continue;

                cudaStream_t stream = streams[chunk_idx % num_streams];

                // Allocate for this batch: max pairs = row_count_left * right_rows
                size_t max_batch_pairs = row_count_left * right_rows;
                size_t *d_left_idx = nullptr;
                size_t *d_right_idx = nullptr;
                unsigned long long *d_count = nullptr;

                err = cudaMalloc(&d_left_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_left_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_left_idx);

                err = cudaMalloc(&d_right_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_idx);

                err = cudaMalloc(&d_count, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_count for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_count);

                err = cudaMemset(d_count, 0, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to initialize d_count for batch: ") + cudaGetErrorString(err));
                }

                float *d_right_data = nullptr;
                err = cudaMalloc(&d_right_data, right_rows * sizeof(float));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_data (FLOAT): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_data);

                err = cudaMemcpyAsync(d_right_data,
                                      static_cast<float *>(right_table.data[index_right]) + right_start,
                                      right_rows * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy right_table.data to d_right_data (FLOAT): ") + cudaGetErrorString(err));
                }

                size_t chunk_blocks = (std::max(row_count_left, right_rows) + numThreads - 1) / numThreads;
                size_t shared_mem_size2 = numThreads * sizeof(float);
                hashJoinKernel<float><<<chunk_blocks, numThreads, shared_mem_size2, stream>>>(
                    d_left_data, d_right_data,
                    row_count_left, right_rows,
                    d_left_idx, d_right_idx, d_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch hashJoinKernel (FLOAT): ") + cudaGetErrorString(err));
                }

                err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to synchronize stream (FLOAT): ") + cudaGetErrorString(err));
                }

                unsigned long long h_count = 0;
                err = cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_count to h_count (FLOAT): ") + cudaGetErrorString(err));
                }

                if (h_count > max_batch_pairs)
                {
                    throw std::runtime_error("Batch produced more pairs than allocated: " + std::to_string(h_count));
                }

                std::vector<size_t> batch_left_indices(h_count);
                std::vector<size_t> batch_right_indices(h_count);
                if (h_count > 0)
                {
                    err = cudaMemcpy(batch_left_indices.data(), d_left_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_left_idx to batch_left_indices (FLOAT): ") + cudaGetErrorString(err));
                    }
                    err = cudaMemcpy(batch_right_indices.data(), d_right_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_right_idx to batch_right_indices (FLOAT): ") + cudaGetErrorString(err));
                    }

                    // Adjust right_indices for chunk offset
                    for (auto &idx : batch_right_indices)
                    {
                        idx += right_start;
                    }

                    left_indices.insert(left_indices.end(), batch_left_indices.begin(), batch_left_indices.end());
                    right_indices.insert(right_indices.end(), batch_right_indices.begin(), batch_right_indices.end());
                }

                // Free batch-specific allocations
                cudaFree(d_left_idx);
                allocations.pop_back();
                cudaFree(d_right_idx);
                allocations.pop_back();
                cudaFree(d_count);
                allocations.pop_back();
                cudaFree(d_right_data);
                allocations.pop_back();
            }
        }
        else if (left_col.type == DataType::DATETIME)
        {
            uint64_t *d_left_data = nullptr;
            cudaError_t err = cudaMalloc(&d_left_data, row_count_left * sizeof(uint64_t));
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to allocate d_left_data (DATETIME): ") + cudaGetErrorString(err));
            }
            allocations.push_back(d_left_data);

            err = cudaMemcpy(d_left_data, left_table.data[index_left], row_count_left * sizeof(uint64_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to copy left_table.data to d_left_data (DATETIME): ") + cudaGetErrorString(err));
            }

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                size_t right_start = chunk_idx * chunk_size;
                size_t right_rows = std::min(chunk_size, row_count_right - right_start);
                if (right_rows == 0)
                    continue;

                cudaStream_t stream = streams[chunk_idx % num_streams];

                size_t max_batch_pairs = row_count_left * right_rows;
                size_t *d_left_idx = nullptr;
                size_t *d_right_idx = nullptr;
                unsigned long long *d_count = nullptr;

                err = cudaMalloc(&d_left_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_left_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_left_idx);

                err = cudaMalloc(&d_right_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_idx);

                err = cudaMalloc(&d_count, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_count for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_count);

                err = cudaMemset(d_count, 0, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to initialize d_count for batch: ") + cudaGetErrorString(err));
                }

                uint64_t *d_right_data = nullptr;
                err = cudaMalloc(&d_right_data, right_rows * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_data (DATETIME): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_data);

                err = cudaMemcpyAsync(d_right_data,
                                      static_cast<uint64_t *>(right_table.data[index_right]) + right_start,
                                      right_rows * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy right_table.data to d_right_data (DATETIME): ") + cudaGetErrorString(err));
                }

                size_t chunk_blocks = (std::max(row_count_left, right_rows) + numThreads - 1) / numThreads;
                hashJoinKernel<uint64_t><<<chunk_blocks, numThreads, shared_mem_size, stream>>>(
                    d_left_data, d_right_data,
                    row_count_left, right_rows,
                    d_left_idx, d_right_idx, d_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch hashJoinKernel (DATETIME): ") + cudaGetErrorString(err));
                }

                err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to synchronize stream (DATETIME): ") + cudaGetErrorString(err));
                }

                unsigned long long h_count = 0;
                err = cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_count to h_count (DATETIME): ") + cudaGetErrorString(err));
                }

                if (h_count > max_batch_pairs)
                {
                    throw std::runtime_error("Batch produced more pairs than allocated: " + std::to_string(h_count));
                }

                std::vector<size_t> batch_left_indices(h_count);
                std::vector<size_t> batch_right_indices(h_count);
                if (h_count > 0)
                {
                    err = cudaMemcpy(batch_left_indices.data(), d_left_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_left_idx to batch_left_indices (DATETIME): ") + cudaGetErrorString(err));
                    }
                    err = cudaMemcpy(batch_right_indices.data(), d_right_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_right_idx to batch_right_indices (DATETIME): ") + cudaGetErrorString(err));
                    }

                    for (auto &idx : batch_right_indices)
                    {
                        idx += right_start;
                    }

                    left_indices.insert(left_indices.end(), batch_left_indices.begin(), batch_left_indices.end());
                    right_indices.insert(right_indices.end(), batch_right_indices.begin(), batch_right_indices.end());
                }

                cudaFree(d_left_idx);
                allocations.pop_back();
                cudaFree(d_right_idx);
                allocations.pop_back();
                cudaFree(d_count);
                allocations.pop_back();
                cudaFree(d_right_data);
                allocations.pop_back();
            }
        }
        else if (left_col.type == DataType::STRING)
        {
            const char **d_left_data = nullptr;
            cudaError_t err = cudaMalloc(&d_left_data, row_count_left * sizeof(char *));
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to allocate d_left_data (STRING): ") + cudaGetErrorString(err));
            }
            allocations.push_back(d_left_data);

            std::vector<char *> d_strings_left(row_count_left);
            const char **host_strings_left = static_cast<const char **>(left_table.data[index_left]);
            for (size_t i = 0; i < row_count_left; ++i)
            {
                size_t len = strlen(host_strings_left[i]) + 1;
                err = cudaMalloc(&d_strings_left[i], len);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_strings_left[i] (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_strings_left[i]);
                err = cudaMemcpy(d_strings_left[i], host_strings_left[i], len, cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy host_strings_left[i] to d_strings_left[i] (STRING): ") + cudaGetErrorString(err));
                }
                err = cudaMemcpy(&d_left_data[i], &d_strings_left[i], sizeof(char *), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_strings_left[i] to d_left_data[i] (STRING): ") + cudaGetErrorString(err));
                }
            }

            for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                size_t right_start = chunk_idx * chunk_size;
                size_t right_rows = std::min(chunk_size, row_count_right - right_start);
                if (right_rows == 0)
                    continue;

                cudaStream_t stream = streams[chunk_idx % num_streams];

                size_t max_batch_pairs = row_count_left * right_rows;
                size_t *d_left_idx = nullptr;
                size_t *d_right_idx = nullptr;
                unsigned long long *d_count = nullptr;

                err = cudaMalloc(&d_left_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_left_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_left_idx);

                err = cudaMalloc(&d_right_idx, max_batch_pairs * sizeof(size_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_idx for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_idx);

                err = cudaMalloc(&d_count, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_count for batch: ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_count);

                err = cudaMemset(d_count, 0, sizeof(unsigned long long));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to initialize d_count for batch: ") + cudaGetErrorString(err));
                }

                const char **d_right_data = nullptr;
                std::vector<char *> d_strings_right(right_rows);
                err = cudaMalloc(&d_right_data, right_rows * sizeof(char *));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_right_data (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_right_data);

                const char **host_strings_right = static_cast<const char **>(right_table.data[index_right]);
                for (size_t i = 0; i < right_rows; ++i)
                {
                    size_t idx = right_start + i;
                    size_t len = strlen(host_strings_right[idx]) + 1;
                    err = cudaMalloc(&d_strings_right[i], len);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to allocate d_strings_right[i] (STRING): ") + cudaGetErrorString(err));
                    }
                    allocations.push_back(d_strings_right[i]);
                    err = cudaMemcpyAsync(d_strings_right[i], host_strings_right[idx], len, cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy host_strings_right[idx] to d_strings_right[i] (STRING): ") + cudaGetErrorString(err));
                    }
                    err = cudaMemcpyAsync(&d_right_data[i], &d_strings_right[i], sizeof(char *), cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_strings_right[i] to d_right_data[i] (STRING): ") + cudaGetErrorString(err));
                    }
                }

                size_t chunk_blocks = (std::max(row_count_left, right_rows) + numThreads - 1) / numThreads;
                hashJoinKernel<const char *><<<chunk_blocks, numThreads, shared_mem_size, stream>>>(
                    d_left_data, d_right_data,
                    row_count_left, right_rows,
                    d_left_idx, d_right_idx, d_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch hashJoinKernel (STRING): ") + cudaGetErrorString(err));
                }

                err = cudaStreamSynchronize(stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to synchronize stream (STRING): ") + cudaGetErrorString(err));
                }

                unsigned long long h_count = 0;
                err = cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_count to h_count (STRING): ") + cudaGetErrorString(err));
                }

                if (h_count > max_batch_pairs)
                {
                    throw std::runtime_error("Batch produced more pairs than allocated: " + std::to_string(h_count));
                }

                std::vector<size_t> batch_left_indices(h_count);
                std::vector<size_t> batch_right_indices(h_count);
                if (h_count > 0)
                {
                    err = cudaMemcpy(batch_left_indices.data(), d_left_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_left_idx to batch_left_indices (STRING): ") + cudaGetErrorString(err));
                    }
                    err = cudaMemcpy(batch_right_indices.data(), d_right_idx, h_count * sizeof(size_t), cudaMemcpyDeviceToHost);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(std::string("Failed to copy d_right_idx to batch_right_indices (STRING): ") + cudaGetErrorString(err));
                    }

                    for (auto &idx : batch_right_indices)
                    {
                        idx += right_start;
                    }

                    left_indices.insert(left_indices.end(), batch_left_indices.begin(), batch_left_indices.end());
                    right_indices.insert(right_indices.end(), batch_right_indices.begin(), batch_right_indices.end());
                }

                cudaFree(d_left_idx);
                allocations.pop_back();
                cudaFree(d_right_idx);
                allocations.pop_back();
                cudaFree(d_count);
                allocations.pop_back();
                cudaFree(d_right_data);
                allocations.pop_back();
                for (auto ptr : d_strings_right)
                {
                    cudaFree(ptr);
                    allocations.pop_back();
                }
            }
        }
        else
        {
            throw std::runtime_error("Unsupported join column type");
        }

        for (auto &stream : streams)
        {
            cudaError_t err = cudaStreamDestroy(stream);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to destroy CUDA stream: ") + cudaGetErrorString(err));
            }
        }
        for (auto ptr : allocations)
        {
            cudaFree(ptr);
        }
    }
    catch (...)
    {
        for (auto ptr : allocations)
        {
            if (ptr)
                cudaFree(ptr);
        }
        for (auto &stream : streams)
        {
            cudaStreamDestroy(stream);
        }
        throw;
    }
}
TableResults HashJoin::executeJoin(const TableResults &left_table, const TableResults &right_table)
{
    if (left_table.row_count == 0 || right_table.row_count == 0)
    {
        TableResults result;
        result.row_count = 0;
        result.has_more = false;
        return result;
    }

    std::vector<size_t> left_indices;
    std::vector<size_t> right_indices;
    getIndexOfSelectedRows(left_table, right_table, left_indices, right_indices);

    TableResults result;
    result.column_count = left_table.columns.size() + right_table.columns.size();
    result.row_count = left_indices.size();
    result.has_more = left_table.has_more;
    result.columns = left_table.columns;
    result.columns.insert(result.columns.end(), right_table.columns.begin(), right_table.columns.end());
    result.data.resize(left_table.columns.size() + right_table.columns.size());

    for (size_t i = 0; i < result.columns.size(); ++i)
    {
        result.columns[i].idx = i;
    }

    size_t *d_left_idx = nullptr;
    size_t *d_right_idx = nullptr;
    std::vector<void *> allocations;
    cudaError_t err = cudaMalloc(&d_left_idx, result.row_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate d_left_idx: ") + cudaGetErrorString(err));
    }
    allocations.push_back(d_left_idx);

    err = cudaMalloc(&d_right_idx, result.row_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate d_right_idx: ") + cudaGetErrorString(err));
    }
    allocations.push_back(d_right_idx);

    err = cudaMemcpy(d_left_idx, left_indices.data(), result.row_count * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to copy left_indices to d_left_idx: ") + cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_right_idx, right_indices.data(), result.row_count * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to copy right_indices to d_right_idx: ") + cudaGetErrorString(err));
    }

    int numThreads = 256;
    int numBlocks = (result.row_count + numThreads - 1) / numThreads;
    size_t total_columns = left_table.columns.size() + right_table.columns.size();
    std::vector<cudaStream_t> streams(total_columns);

    for (auto &stream : streams)
    {
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
        }
    }

    try
    {
        size_t col_idx = 0;
        // Process left table columns
        for (size_t i = 0; i < left_table.columns.size(); ++i)
        {
            cudaStream_t stream = streams[col_idx];
            switch (left_table.columns[i].type)
            {
            case DataType::FLOAT:
            {
                float *d_input = nullptr;
                float *d_output = nullptr;
                err = cudaMalloc(&d_input, left_table.row_count * sizeof(float));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input (FLOAT): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input);
                err = cudaMalloc(&d_output, result.row_count * sizeof(float));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output (FLOAT): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output);
                err = cudaMemcpyAsync(d_input, left_table.data[i], left_table.row_count * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy left_table.data to d_input (FLOAT): ") + cudaGetErrorString(err));
                }
                getRowsKernel<float><<<numBlocks, numThreads, 0, stream>>>(d_input, d_left_idx, d_output, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (FLOAT): ") + cudaGetErrorString(err));
                }
                float *h_output_data = static_cast<float *>(malloc(result.row_count * sizeof(float)));
                result.data[col_idx] = h_output_data;
                err = cudaMemcpyAsync(h_output_data, d_output, result.row_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output to h_output_data (FLOAT): ") + cudaGetErrorString(err));
                }
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t *d_input = nullptr;
                uint64_t *d_output = nullptr;
                err = cudaMalloc(&d_input, left_table.row_count * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input (DATETIME): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input);
                err = cudaMalloc(&d_output, result.row_count * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output (DATETIME): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output);
                err = cudaMemcpyAsync(d_input, left_table.data[i], left_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy left_table.data to d_input (DATETIME): ") + cudaGetErrorString(err));
                }
                getRowsKernel<uint64_t><<<numBlocks, numThreads, 0, stream>>>(d_input, d_left_idx, d_output, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (DATETIME): ") + cudaGetErrorString(err));
                }
                uint64_t *h_output_data = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
                result.data[col_idx] = h_output_data;
                err = cudaMemcpyAsync(h_output_data, d_output, result.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output to h_output_data (DATETIME): ") + cudaGetErrorString(err));
                }
                break;
            }
            case DataType::STRING:
            {
                const char **h_input_strings = static_cast<const char **>(left_table.data[i]);
                const char **h_output_strings = static_cast<const char **>(malloc(result.row_count * sizeof(char *)));
                result.data[col_idx] = h_output_strings;

                const char **d_input_strings = nullptr;
                const char **d_output_strings = nullptr;
                err = cudaMalloc(&d_input_strings, left_table.row_count * sizeof(char *));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input_strings (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input_strings);
                err = cudaMalloc(&d_output_strings, result.row_count * sizeof(char *));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output_strings (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output_strings);
                err = cudaMemcpyAsync(d_input_strings, h_input_strings, left_table.row_count * sizeof(char *), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy h_input_strings to d_input_strings (STRING): ") + cudaGetErrorString(err));
                }
                getRowsKernel<const char *><<<numBlocks, numThreads, 0, stream>>>(d_input_strings, d_left_idx, d_output_strings, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (STRING): ") + cudaGetErrorString(err));
                }
                err = cudaMemcpyAsync(h_output_strings, d_output_strings, result.row_count * sizeof(char *), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output_strings to h_output_strings (STRING): ") + cudaGetErrorString(err));
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(left_table.columns[i].type)));
            }
            col_idx++;
        }

        // Process right table columns
        for (size_t i = 0; i < right_table.columns.size(); ++i)
        {
            cudaStream_t stream = streams[col_idx];
            switch (right_table.columns[i].type)
            {
            case DataType::FLOAT:
            {
                float *d_input = nullptr;
                float *d_output = nullptr;
                err = cudaMalloc(&d_input, right_table.row_count * sizeof(float));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input (FLOAT): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input);
                err = cudaMalloc(&d_output, result.row_count * sizeof(float));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output (FLOAT): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output);
                err = cudaMemcpyAsync(d_input, right_table.data[i], right_table.row_count * sizeof(float), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy right_table.data to d_input (FLOAT): ") + cudaGetErrorString(err));
                }
                getRowsKernel<float><<<numBlocks, numThreads, 0, stream>>>(d_input, d_right_idx, d_output, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (FLOAT): ") + cudaGetErrorString(err));
                }
                float *h_output_data = static_cast<float *>(malloc(result.row_count * sizeof(float)));
                result.data[col_idx] = h_output_data;
                err = cudaMemcpyAsync(h_output_data, d_output, result.row_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output to h_output_data (FLOAT): ") + cudaGetErrorString(err));
                }
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t *d_input = nullptr;
                uint64_t *d_output = nullptr;
                err = cudaMalloc(&d_input, right_table.row_count * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input (DATETIME): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input);
                err = cudaMalloc(&d_output, result.row_count * sizeof(uint64_t));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output (DATETIME): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output);
                err = cudaMemcpyAsync(d_input, right_table.data[i], right_table.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy right_table.data to d_input (DATETIME): ") + cudaGetErrorString(err));
                }
                getRowsKernel<uint64_t><<<numBlocks, numThreads, 0, stream>>>(d_input, d_right_idx, d_output, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (DATETIME): ") + cudaGetErrorString(err));
                }
                uint64_t *h_output_data = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
                result.data[col_idx] = h_output_data;
                err = cudaMemcpyAsync(h_output_data, d_output, result.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output to h_output_data (DATETIME): ") + cudaGetErrorString(err));
                }
                break;
            }
            case DataType::STRING:
            {
                const char **h_input_strings = static_cast<const char **>(right_table.data[i]);
                const char **h_output_strings = static_cast<const char **>(malloc(result.row_count * sizeof(char *)));
                result.data[col_idx] = h_output_strings;

                const char **d_input_strings = nullptr;
                const char **d_output_strings = nullptr;
                err = cudaMalloc(&d_input_strings, right_table.row_count * sizeof(char *));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_input_strings (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_input_strings);
                err = cudaMalloc(&d_output_strings, result.row_count * sizeof(char *));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to allocate d_output_strings (STRING): ") + cudaGetErrorString(err));
                }
                allocations.push_back(d_output_strings);
                err = cudaMemcpyAsync(d_input_strings, h_input_strings, right_table.row_count * sizeof(char *), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy h_input_strings to d_input_strings (STRING): ") + cudaGetErrorString(err));
                }
                getRowsKernel<const char *><<<numBlocks, numThreads, 0, stream>>>(d_input_strings, d_right_idx, d_output_strings, result.row_count);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to launch getRowsKernel (STRING): ") + cudaGetErrorString(err));
                }
                err = cudaMemcpyAsync(h_output_strings, d_output_strings, result.row_count * sizeof(char *), cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(std::string("Failed to copy d_output_strings to h_output_strings (STRING): ") + cudaGetErrorString(err));
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(right_table.columns[i].type)));
            }
            col_idx++;
        }

        for (auto &stream : streams)
        {
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to synchronize stream: ") + cudaGetErrorString(err));
            }
            err = cudaStreamDestroy(stream);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to destroy CUDA stream: ") + cudaGetErrorString(err));
            }
        }
        for (auto ptr : allocations)
        {
            cudaFree(ptr);
        }
    }
    catch (...)
    {
        for (auto ptr : allocations)
        {
            if (ptr)
                cudaFree(ptr);
        }
        for (auto &stream : streams)
        {
            cudaStreamDestroy(stream);
        }
        throw;
    }

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
