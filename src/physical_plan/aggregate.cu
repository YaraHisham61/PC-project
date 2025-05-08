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
    if (input.row_count == 0)
    {
        return input;
    }

    TableResults result;
    result.row_count = 1;
    result.has_more = input.has_more;
    result.total_rows += input.row_count;
    result.column_count = aggregates.size();
    result.data.resize(result.column_count);

    const size_t chunk_size = 10000;
    size_t num_chunks = (input.row_count + chunk_size - 1) / chunk_size;
    std::vector<cudaStream_t> streams(std::max<size_t>(1, num_chunks));
    for (auto &stream : streams)
    {
        checkCudaError(cudaStreamCreate(&stream), "Failed to create CUDA stream", __FILE__, __LINE__);
    }

    std::vector<void *> allocations;
    try
    {
        int numThreads = 256;

        for (size_t i = 0; i < aggregates.size(); ++i)
        {
            size_t shared_mem_size = ((numThreads + 31) / 32) * getDataTypeNumBytes(input.columns[aggregates[i].column_index].type);
            ColumnInfo col;
            col.name = getAggregateName(aggregates[i], input);
            col.type = getOutputType(aggregates[i], input);
            col.idx = aggregates[i].column_index;
            result.columns.push_back(col);

            switch (aggregates[i].type)
            {
            case AggregateType::COUNT_STAR:
            {
                result.data[i] = new float(input.row_count);
                break;
            }
            case AggregateType::COUNT:
            {
                float total_count = 0.0f;
                switch (input.columns[aggregates[i].column_index].type)
                {
                case DataType::FLOAT:
                {
                    std::vector<float> partial_counts(num_chunks, 0.0f);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        float *d_input_f = nullptr;
                        float *d_output_f = nullptr;
                        checkCudaError(cudaMalloc(&d_input_f, chunk_rows * sizeof(float)), "Failed to allocate d_input_f (FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_input_f);
                        checkCudaError(cudaMalloc(&d_output_f, sizeof(float)), "Failed to allocate d_output_f (FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_output_f);
                        checkCudaError(cudaMemsetAsync(d_output_f, 0, sizeof(float), streams[stream_idx]), "Failed to memset d_output_f (FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_f, static_cast<float *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_f (FLOAT)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        countElements<float><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_f, d_output_f, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch countElements (FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_counts[chunk_idx], d_output_f, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_f to partial_counts (FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_f), "Failed to free d_input_f (FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_f), "Failed to free d_output_f (FLOAT)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (float count : partial_counts)
                    {
                        total_count += count;
                    }
                    result.data[i] = new float(total_count);
                    break;
                }
                case DataType::DATETIME:
                {
                    std::vector<float> partial_counts(num_chunks, 0.0f);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        uint64_t *d_input_ui = nullptr;
                        float *d_output_f = nullptr;
                        checkCudaError(cudaMalloc(&d_input_ui, chunk_rows * sizeof(uint64_t)), "Failed to allocate d_input_ui (DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_input_ui);
                        checkCudaError(cudaMalloc(&d_output_f, sizeof(float)), "Failed to allocate d_output_f (DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_output_f);
                        checkCudaError(cudaMemsetAsync(d_output_f, 0, sizeof(float), streams[stream_idx]), "Failed to memset d_output_f (DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_ui, static_cast<uint64_t *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_ui (DATETIME)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        countElements<uint64_t><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_ui, d_output_f, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch countElements (DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_counts[chunk_idx], d_output_f, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_f to partial_counts (DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_ui), "Failed to free d_input_ui (DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_f), "Failed to free d_output_f (DATETIME)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (float count : partial_counts)
                    {
                        total_count += count;
                    }
                    result.data[i] = new float(total_count);
                    break;
                }
                case DataType::STRING:
                {
                    std::vector<float> partial_counts(num_chunks, 0.0f);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        char **d_input = nullptr;
                        float *d_output = nullptr;
                        checkCudaError(cudaMalloc(&d_input, chunk_rows * sizeof(char *)), "Failed to allocate d_input (STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_input);
                        checkCudaError(cudaMalloc(&d_output, sizeof(float)), "Failed to allocate d_output (STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_output);
                        checkCudaError(cudaMemsetAsync(d_output, 0, sizeof(float), streams[stream_idx]), "Failed to memset d_output (STRING)", __FILE__, __LINE__);

                        char **host_pointers = new char *[chunk_rows];
                        const char **input_column = static_cast<const char **>(input.data[aggregates[i].column_index]) + chunk_offset;

                        std::vector<void *> string_allocs(chunk_rows);
                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            if (input_column[j])
                            {
                                size_t len = strlen(input_column[j]) + 1;
                                checkCudaError(cudaMalloc(&host_pointers[j], len), "Failed to allocate host_pointers[j] (STRING)", __FILE__, __LINE__);
                                string_allocs[j] = host_pointers[j];
                                checkCudaError(cudaMemcpyAsync(host_pointers[j], input_column[j], len, cudaMemcpyHostToDevice, streams[stream_idx]),
                                               "Failed to copy input_column[j] to host_pointers[j] (STRING)", __FILE__, __LINE__);
                            }
                            else
                            {
                                host_pointers[j] = nullptr;
                            }
                        }

                        checkCudaError(cudaMemcpyAsync(d_input, host_pointers, chunk_rows * sizeof(char *), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy host_pointers to d_input (STRING)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        countElements<char *><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input, d_output, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch countElements (STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_counts[chunk_idx], d_output, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output to partial_counts (STRING)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            if (host_pointers[j])
                            {
                                checkCudaError(cudaFree(host_pointers[j]), "Failed to free host_pointers[j] (STRING)", __FILE__, __LINE__);
                                string_allocs[j] = nullptr;
                            }
                        }
                        delete[] host_pointers;
                        checkCudaError(cudaFree(d_input), "Failed to free d_input (STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output), "Failed to free d_output (STRING)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (float count : partial_counts)
                    {
                        total_count += count;
                    }
                    result.data[i] = new float(total_count);
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported data type for COUNT aggregate");
                }
                break;
            }
            case AggregateType::AVG:
            case AggregateType::SUM:
            {
                if (input.columns[aggregates[i].column_index].type != DataType::FLOAT)
                {
                    throw std::runtime_error("SUM only supported for FLOAT");
                }
                float total_sum = 0.0f;
                std::vector<float> partial_sums(num_chunks, 0.0f);
                for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                {
                    size_t chunk_offset = chunk_idx * chunk_size;
                    size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                    size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                    float *d_input_f = nullptr;
                    float *d_output_f = nullptr;
                    checkCudaError(cudaMalloc(&d_input_f, chunk_rows * sizeof(float)), "Failed to allocate d_input_f (SUM)", __FILE__, __LINE__);
                    allocations.push_back(d_input_f);
                    checkCudaError(cudaMalloc(&d_output_f, sizeof(float)), "Failed to allocate d_output_f (SUM)", __FILE__, __LINE__);
                    allocations.push_back(d_output_f);
                    checkCudaError(cudaMemsetAsync(d_output_f, 0, sizeof(float), streams[stream_idx]), "Failed to memset d_output_f (SUM)", __FILE__, __LINE__);

                    checkCudaError(cudaMemcpyAsync(d_input_f, static_cast<float *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                   chunk_rows * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                   "Failed to copy input to d_input_f (SUM)", __FILE__, __LINE__);

                    int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                    findSumElement<<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_f, d_output_f, chunk_rows);
                    checkCudaError(cudaGetLastError(), "Failed to launch findSumElement", __FILE__, __LINE__);
                    checkCudaError(cudaMemcpyAsync(&partial_sums[chunk_idx], d_output_f, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                   "Failed to copy d_output_f to partial_sums (SUM)", __FILE__, __LINE__);

                    checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                    checkCudaError(cudaFree(d_input_f), "Failed to free d_input_f (SUM)", __FILE__, __LINE__);
                    checkCudaError(cudaFree(d_output_f), "Failed to free d_output_f (SUM)", __FILE__, __LINE__);
                    allocations.pop_back();
                    allocations.pop_back();
                }
                for (float sum : partial_sums)
                {
                    total_sum += sum;
                }
                result.data[i] = new float(total_sum);
                break;
            }

            case AggregateType::MIN:
            {
                switch (input.columns[aggregates[i].column_index].type)
                {
                case DataType::FLOAT:
                {
                    float min_value = FLT_MAX;
                    std::vector<float> partial_mins(num_chunks, FLT_MAX);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        float *d_input_f = nullptr;
                        float *d_output_f = nullptr;
                        checkCudaError(cudaMalloc(&d_input_f, chunk_rows * sizeof(float)), "Failed to allocate d_input_f (MIN FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_input_f);
                        checkCudaError(cudaMalloc(&d_output_f, sizeof(float)), "Failed to allocate d_output_f (MIN FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_output_f);
                        checkCudaError(cudaMemcpyAsync(d_output_f, &partial_mins[chunk_idx], sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy partial_mins to d_output_f (MIN FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_f, static_cast<float *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_f (MIN FLOAT)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMinElement<float><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_f, d_output_f, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMinElement (MIN FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_mins[chunk_idx], d_output_f, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_f to partial_mins (MIN FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_f), "Failed to free d_input_f (MIN FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_f), "Failed to free d_output_f (MIN FLOAT)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (float min : partial_mins)
                    {
                        if (min < min_value)
                            min_value = min;
                    }
                    result.data[i] = new float(min_value);
                    break;
                }
                case DataType::DATETIME:
                {
                    uint64_t min_value = UINT64_MAX;
                    std::vector<uint64_t> partial_mins(num_chunks, UINT64_MAX);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        uint64_t *d_input_ui = nullptr;
                        uint64_t *d_output_ui = nullptr;
                        checkCudaError(cudaMalloc(&d_input_ui, chunk_rows * sizeof(uint64_t)), "Failed to allocate d_input_ui (MIN DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_input_ui);
                        checkCudaError(cudaMalloc(&d_output_ui, sizeof(uint64_t)), "Failed to allocate d_output_ui (MIN DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_output_ui);
                        checkCudaError(cudaMemcpyAsync(d_output_ui, &partial_mins[chunk_idx], sizeof(uint64_t), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy partial_mins to d_output_ui (MIN DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_ui, static_cast<uint64_t *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_ui (MIN DATETIME)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMinElement<uint64_t><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_ui, d_output_ui, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMinElement (MIN DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_mins[chunk_idx], d_output_ui, sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_ui to partial_mins (MIN DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_ui), "Failed to free d_input_ui (MIN DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_ui), "Failed to free d_output_ui (MIN DATETIME)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (uint64_t min : partial_mins)
                    {
                        if (min < min_value)
                            min_value = min;
                    }
                    result.data[i] = new uint64_t(min_value);
                    break;
                }
                case DataType::STRING:
                {
                    std::vector<char *> partial_mins(num_chunks, nullptr);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        char **d_input_char = nullptr;
                        char **d_output_char = nullptr;
                        checkCudaError(cudaMalloc(&d_input_char, chunk_rows * sizeof(char *)), "Failed to allocate d_input_char (MIN STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_input_char);
                        checkCudaError(cudaMalloc(&d_output_char, sizeof(char *)), "Failed to allocate d_output_char (MIN STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_output_char);

                        char **d_strings = new char *[chunk_rows];
                        std::vector<void *> string_allocs(chunk_rows);
                        const char **host_strings = static_cast<const char **>(input.data[aggregates[i].column_index]) + chunk_offset;

                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            size_t len = strlen(host_strings[j]) + 1;
                            checkCudaError(cudaMalloc(&d_strings[j], len), "Failed to allocate d_strings[j] (MIN STRING)", __FILE__, __LINE__);
                            string_allocs[j] = d_strings[j];
                            checkCudaError(cudaMemcpyAsync(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice, streams[stream_idx]),
                                           "Failed to copy host_strings[j] to d_strings[j] (MIN STRING)", __FILE__, __LINE__);
                            checkCudaError(cudaMemcpyAsync(&d_input_char[j], &d_strings[j], sizeof(char *), cudaMemcpyHostToDevice, streams[stream_idx]),
                                           "Failed to copy d_strings[j] to d_input_char[j] (MIN STRING)", __FILE__, __LINE__);
                        }

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMinElement<char *><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_char, d_output_char, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMinElement (MIN STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_mins[chunk_idx], d_output_char, sizeof(char *), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_char to partial_mins (MIN STRING)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            checkCudaError(cudaFree(d_strings[j]), "Failed to free d_strings[j] (MIN STRING)", __FILE__, __LINE__);
                            string_allocs[j] = nullptr;
                        }
                        delete[] d_strings;
                        checkCudaError(cudaFree(d_input_char), "Failed to free d_input_char (MIN STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_char), "Failed to free d_output_char (MIN STRING)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }

                    char *min_string = nullptr;
                    size_t min_len = 0;
                    for (char *str : partial_mins)
                    {
                        if (!str)
                            continue;
                        char *temp = new char[chunk_size];
                        size_t len = 0;
                        char c;
                        do
                        {
                            checkCudaError(cudaMemcpy(&c, str + len, 1, cudaMemcpyDeviceToHost), "Failed to copy string char (MIN STRING)", __FILE__, __LINE__);
                            temp[len] = c;
                            len++;
                        } while (c != '\0' && len < chunk_size);
                        if (!min_string || strcmp(temp, min_string) < 0)
                        {
                            delete[] min_string;
                            min_string = temp;
                            min_len = len;
                        }
                        else
                        {
                            delete[] temp;
                        }
                    }

                    if (min_string)
                    {
                        char *h_output_string = new char[min_len];
                        memcpy(h_output_string, min_string, min_len);
                        delete[] min_string;
                        result.data[i] = new char *[1];
                        static_cast<char **>(result.data[i])[0] = h_output_string;
                    }
                    else
                    {
                        result.data[i] = new char *[1];
                        static_cast<char **>(result.data[i])[0] = new char[1]{'\0'};
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported data type for MIN aggregate");
                }
                break;
            }
            case AggregateType::MAX:
            {
                switch (input.columns[aggregates[i].column_index].type)
                {
                case DataType::FLOAT:
                {
                    float max_value = -FLT_MAX;
                    std::vector<float> partial_maxs(num_chunks, -FLT_MAX);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        float *d_input_f = nullptr;
                        float *d_output_f = nullptr;
                        checkCudaError(cudaMalloc(&d_input_f, chunk_rows * sizeof(float)), "Failed to allocate d_input_f (MAX FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_input_f);
                        checkCudaError(cudaMalloc(&d_output_f, sizeof(float)), "Failed to allocate d_output_f (MAX FLOAT)", __FILE__, __LINE__);
                        allocations.push_back(d_output_f);
                        checkCudaError(cudaMemcpyAsync(d_output_f, &partial_maxs[chunk_idx], sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy partial_maxs to d_output_f (MAX FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_f, static_cast<float *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(float), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_f (MAX FLOAT)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMaxElement<float><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_f, d_output_f, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMaxElement (MAX FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_maxs[chunk_idx], d_output_f, sizeof(float), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_f to partial_maxs (MAX FLOAT)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_f), "Failed to free d_input_f (MAX FLOAT)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_f), "Failed to free d_output_f (MAX FLOAT)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (float max : partial_maxs)
                    {
                        if (max > max_value)
                            max_value = max;
                    }
                    result.data[i] = new float(max_value);
                    break;
                }
                case DataType::DATETIME:
                {
                    uint64_t max_value = 0;
                    std::vector<uint64_t> partial_maxs(num_chunks, 0);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        uint64_t *d_input_ui = nullptr;
                        uint64_t *d_output_ui = nullptr;
                        checkCudaError(cudaMalloc(&d_input_ui, chunk_rows * sizeof(uint64_t)), "Failed to allocate d_input_ui (MAX DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_input_ui);
                        checkCudaError(cudaMalloc(&d_output_ui, sizeof(uint64_t)), "Failed to allocate d_output_ui (MAX DATETIME)", __FILE__, __LINE__);
                        allocations.push_back(d_output_ui);
                        checkCudaError(cudaMemcpyAsync(d_output_ui, &partial_maxs[chunk_idx], sizeof(uint64_t), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy partial_maxs to d_output_ui (MAX DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaMemcpyAsync(d_input_ui, static_cast<uint64_t *>(input.data[aggregates[i].column_index]) + chunk_offset,
                                                       chunk_rows * sizeof(uint64_t), cudaMemcpyHostToDevice, streams[stream_idx]),
                                       "Failed to copy input to d_input_ui (MAX DATETIME)", __FILE__, __LINE__);

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMaxElement<uint64_t><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_ui, d_output_ui, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMaxElement (MAX DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_maxs[chunk_idx], d_output_ui, sizeof(uint64_t), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_ui to partial_maxs (MAX DATETIME)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_input_ui), "Failed to free d_input_ui (MAX DATETIME)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_ui), "Failed to free d_output_ui (MAX DATETIME)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }
                    for (uint64_t max : partial_maxs)
                    {
                        if (max > max_value)
                            max_value = max;
                    }
                    result.data[i] = new uint64_t(max_value);
                    break;
                }
                case DataType::STRING:
                {
                    std::vector<char *> partial_maxs(num_chunks, nullptr);
                    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
                    {
                        size_t chunk_offset = chunk_idx * chunk_size;
                        size_t chunk_rows = std::min(chunk_size, input.row_count - chunk_offset);
                        size_t stream_idx = num_chunks > 1 ? chunk_idx : 0;

                        char **d_input_char = nullptr;
                        char **d_output_char = nullptr;
                        checkCudaError(cudaMalloc(&d_input_char, chunk_rows * sizeof(char *)), "Failed to allocate d_input_char (MAX STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_input_char);
                        checkCudaError(cudaMalloc(&d_output_char, sizeof(char *)), "Failed to allocate d_output_char (MAX STRING)", __FILE__, __LINE__);
                        allocations.push_back(d_output_char);

                        char **d_strings = new char *[chunk_rows];
                        std::vector<void *> string_allocs(chunk_rows);
                        const char **host_strings = static_cast<const char **>(input.data[aggregates[i].column_index]) + chunk_offset;

                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            size_t len = strlen(host_strings[j]) + 1;
                            checkCudaError(cudaMalloc(&d_strings[j], len), "Failed to allocate d_strings[j] (MAX STRING)", __FILE__, __LINE__);
                            string_allocs[j] = d_strings[j];
                            checkCudaError(cudaMemcpyAsync(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice, streams[stream_idx]),
                                           "Failed to copy host_strings[j] to d_strings[j] (MAX STRING)", __FILE__, __LINE__);
                            checkCudaError(cudaMemcpyAsync(&d_input_char[j], &d_strings[j], sizeof(char *), cudaMemcpyHostToDevice, streams[stream_idx]),
                                           "Failed to copy d_strings[j] to d_input_char[j] (MAX STRING)", __FILE__, __LINE__);
                        }

                        int numBlocks = (chunk_rows + numThreads - 1) / numThreads;
                        findMaxElement<char *><<<numBlocks, numThreads, shared_mem_size, streams[stream_idx]>>>(d_input_char, d_output_char, chunk_rows);
                        checkCudaError(cudaGetLastError(), "Failed to launch findMaxElement (MAX STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaMemcpyAsync(&partial_maxs[chunk_idx], d_output_char, sizeof(char *), cudaMemcpyDeviceToHost, streams[stream_idx]),
                                       "Failed to copy d_output_char to partial_maxs (MAX STRING)", __FILE__, __LINE__);

                        checkCudaError(cudaStreamSynchronize(streams[stream_idx]), "Failed to synchronize stream", __FILE__, __LINE__);
                        for (size_t j = 0; j < chunk_rows; j++)
                        {
                            checkCudaError(cudaFree(d_strings[j]), "Failed to free d_strings[j] (MAX STRING)", __FILE__, __LINE__);
                            string_allocs[j] = nullptr;
                        }
                        delete[] d_strings;
                        checkCudaError(cudaFree(d_input_char), "Failed to free d_input_char (MAX STRING)", __FILE__, __LINE__);
                        checkCudaError(cudaFree(d_output_char), "Failed to free d_output_char (MAX STRING)", __FILE__, __LINE__);
                        allocations.pop_back();
                        allocations.pop_back();
                    }

                    char *max_string = nullptr;
                    size_t max_len = 0;
                    for (char *str : partial_maxs)
                    {
                        if (!str)
                            continue;
                        char *temp = new char[chunk_size];
                        size_t len = 0;
                        char c;
                        do
                        {
                            checkCudaError(cudaMemcpy(&c, str + len, 1, cudaMemcpyDeviceToHost), "Failed to copy string char (MAX STRING)", __FILE__, __LINE__);
                            temp[len] = c;
                            len++;
                        } while (c != '\0' && len < chunk_size);
                        if (!max_string || strcmp(temp, max_string) > 0)
                        {
                            delete[] max_string;
                            max_string = temp;
                            max_len = len;
                        }
                        else
                        {
                            delete[] temp;
                        }
                    }

                    if (max_string)
                    {
                        char *h_output_string = new char[max_len];
                        memcpy(h_output_string, max_string, max_len);
                        delete[] max_string;
                        result.data[i] = new char *[1];
                        static_cast<char **>(result.data[i])[0] = h_output_string;
                    }
                    else
                    {
                        result.data[i] = new char *[1];
                        static_cast<char **>(result.data[i])[0] = new char[1]{'\0'};
                    }
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported data type for MAX aggregate");
                }
                break;
            }
            default:
                throw std::runtime_error("Unsupported aggregate type");
            }
        }

        for (auto &stream : streams)
        {
            checkCudaError(cudaStreamSynchronize(stream), "Failed to synchronize stream", __FILE__, __LINE__);
            checkCudaError(cudaStreamDestroy(stream), "Failed to destroy CUDA stream", __FILE__, __LINE__);
        }
        allocations.clear();
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
        return DataType::FLOAT;
    case AggregateType::COUNT:
        return DataType::FLOAT;
    case AggregateType::SUM:
        return input.columns[agg.column_index].type;
    case AggregateType::AVG:
        return input.columns[agg.column_index].type;
    case AggregateType::MIN:
        return input.columns[agg.column_index].type;
    case AggregateType::MAX:
        return input.columns[agg.column_index].type;
    default:
        throw std::runtime_error("Unsupported aggregate type");
        break;
    }
    return DataType::FLOAT;
}

void Aggregate::updateAggregates(const TableResults &input)
{
    if (!this->intermidiate_results)
    {
        this->intermidiate_results = new TableResults(input);
        return;
    }
    // Update the intermediate results with the new input
    for (size_t i = 0; i < aggregates.size(); ++i)
    {
        switch (aggregates[i].type)
        {
        case AggregateType::COUNT_STAR:
        case AggregateType::COUNT:
        {
            float h_input = static_cast<float *>(input.data[i])[0];
            float h_intermediate = static_cast<float *>((this->intermidiate_results)->data[i])[0];
            static_cast<float *>((this->intermidiate_results)->data[i])[0] = h_intermediate + h_input;
            break;
        }
        case AggregateType::SUM:
        {
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float h_input_sum = static_cast<float *>(input.data[aggregates[i].column_index])[0];
                float h_intermediate_sum = static_cast<float *>((this->intermidiate_results)->data[i])[0];
                static_cast<float *>((this->intermidiate_results)->data[i])[0] = h_intermediate_sum + h_input_sum;
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t h_input_datetime = static_cast<uint64_t *>(input.data[aggregates[i].column_index])[0];
                uint64_t h_intermediate_datetime = static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0];
                h_intermediate_datetime += h_input_datetime;
                static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0] = h_intermediate_datetime;
                break;
            }
            }
            break;
        }

        case AggregateType::AVG:
        {
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float h_input_avg = static_cast<float *>(input.data[aggregates[i].column_index])[0];
                float h_intermediate_avg = static_cast<float *>((this->intermidiate_results)->data[i])[0];
                h_intermediate_avg += h_input_avg;
                static_cast<float *>((this->intermidiate_results)->data[i])[0] = h_intermediate_avg;
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t h_input_avg_datetime = static_cast<uint64_t *>(input.data[aggregates[i].column_index])[0];
                uint64_t h_intermediate_avg_datetime = static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0];
                h_intermediate_avg_datetime += h_input_avg_datetime;
                static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0] = h_intermediate_avg_datetime;
                break;
            }
            }
            break;
        }
        case AggregateType::MIN:
        {
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float h_input_min = static_cast<float *>(input.data[aggregates[i].column_index])[0];
                float h_intermediate_min = static_cast<float *>((this->intermidiate_results)->data[i])[0];
                if (h_input_min < h_intermediate_min)
                {
                    static_cast<float *>((this->intermidiate_results)->data[i])[0] = h_input_min;
                }
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t h_input_min_datetime = static_cast<uint64_t *>(input.data[aggregates[i].column_index])[0];
                uint64_t h_intermediate_min_datetime = static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0];
                if (h_input_min_datetime < h_intermediate_min_datetime)
                {
                    static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0] = h_input_min_datetime;
                }
                break;
            }
            case DataType::STRING:
            {
                char *h_input_min_string = static_cast<char **>(input.data[aggregates[i].column_index])[0];
                char *h_intermediate_min_string = static_cast<char **>((this->intermidiate_results)->data[i])[0];
                if (strcmp(h_input_min_string, h_intermediate_min_string) < 0)
                {
                    static_cast<char **>((this->intermidiate_results)->data[i])[0] = h_input_min_string;
                }
                break;
            }
            }
            break;
        }
        case AggregateType::MAX:
        {
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float h_input_max = static_cast<float *>(input.data[aggregates[i].column_index])[0];
                float h_intermediate_max = static_cast<float *>((this->intermidiate_results)->data[i])[0];
                if (h_input_max > h_intermediate_max)
                {
                    static_cast<float *>((this->intermidiate_results)->data[i])[0] = h_input_max;
                }
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t h_input_max_datetime = static_cast<uint64_t *>(input.data[aggregates[i].column_index])[0];
                uint64_t h_intermediate_max_datetime = static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0];
                if (h_input_max_datetime > h_intermediate_max_datetime)
                {
                    static_cast<uint64_t *>((this->intermidiate_results)->data[i])[0] = h_input_max_datetime;
                }
                break;
            }
            case DataType::STRING:
            {
                char *h_input_max_string = static_cast<char **>(input.data[aggregates[i].column_index])[0];
                char *h_intermediate_max_string = static_cast<char **>((this->intermidiate_results)->data[i])[0];
                if (strcmp(h_input_max_string, h_intermediate_max_string) > 0)
                {
                    static_cast<char **>((this->intermidiate_results)->data[i])[0] = h_input_max_string;
                }
                break;
            }
            }
            break;
        }
        }
    }
}

void Aggregate::finalizeAggregates(TableResults &result) const
{
    for (size_t i = 0; i < aggregates.size(); ++i)
    {
        switch (aggregates[i].type)
        {
        case AggregateType::AVG:
            if (result.data[i])
            {
                float h_intermediate_avg = static_cast<float *>(result.data[i])[0];
                static_cast<float *>(result.data[i])[0] = h_intermediate_avg / result.total_rows;
            }
            break;
        default:
            break;
        }
    }
}
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
