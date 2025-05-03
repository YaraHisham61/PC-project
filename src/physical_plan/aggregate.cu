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
        /*COUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STARCOUNT_STAR*/
        // case AggregateType::COUNT_STAR:
        // {
        //     switch (input.columns[0].type)
        //     {
        //     case DataType::FLOAT:
        //     {
        //         float *d_input = nullptr;
        //         float *d_output = nullptr;
        //         cudaMalloc(&d_input, input.row_count * sizeof(float));
        //         cudaMalloc(&d_output, sizeof(float));
        //         cudaMemset(d_output, 0, sizeof(float));

        //         cudaMemcpy(d_input, input.data[0],
        //                    input.row_count * sizeof(float), cudaMemcpyHostToDevice);

        //         countStar<float><<<numBlocks, numThreads, shared_mem_size>>>(
        //             d_input, d_output, input.row_count);
        //         cudaDeviceSynchronize();

        //         float count_value;
        //         cudaMemcpy(&count_value, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        //         result.data[i] = new float(count_value);

        //         cudaFree(d_input);
        //         cudaFree(d_output);
        //         break;
        //     }
        //     case DataType::DATETIME:
        //     {
        //         uint64_t *d_input = nullptr;
        //         float *d_output = nullptr;
        //         cudaMalloc(&d_input, input.row_count * sizeof(uint64_t));
        //         cudaMalloc(&d_output, sizeof(float));
        //         cudaMemset(d_output, 0, sizeof(float));

        //         cudaMemcpy(d_input, input.data[0],
        //                    input.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

        //         countStar<uint64_t><<<numBlocks, numThreads, shared_mem_size>>>(
        //             d_input, d_output, input.row_count);
        //         cudaDeviceSynchronize();

        //         float count_value;
        //         cudaMemcpy(&count_value, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        //         result.data[i] = new float(count_value);

        //         cudaFree(d_input);
        //         cudaFree(d_output);
        //         break;
        //     }
        //     case DataType::STRING:
        //     {
        //         char **d_input = nullptr;
        //         float *d_output = nullptr;
        //         cudaMalloc(&d_input, input.row_count * sizeof(char *));
        //         cudaMalloc(&d_output, sizeof(float));
        //         cudaMemset(d_output, 0, sizeof(float));

        //         // Only copy the pointers, not string contents
        //         cudaMemcpy(d_input, input.data[0],
        //                    input.row_count * sizeof(char *), cudaMemcpyHostToDevice);

        //         countStar<char *><<<numBlocks, numThreads, shared_mem_size>>>(
        //             d_input, d_output, input.row_count);
        //         cudaDeviceSynchronize();

        //         float count_value;
        //         cudaMemcpy(&count_value, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        //         result.data[i] = new float(count_value);

        //         cudaFree(d_input);
        //         cudaFree(d_output);
        //         break;
        //     }
        //     default:
        //         break;
        //     }
        //     break;
        // }
            /*COUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNTCOUNT*/
        case AggregateType::COUNT:
        {
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float *d_input_f = nullptr;
                float *d_output_f = nullptr;
                cudaMalloc(&d_input_f, input.row_count * sizeof(float));
                cudaMalloc(&d_output_f, sizeof(float));
                cudaMemset(d_output_f, 0, sizeof(float));

                cudaMemcpy(d_input_f, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(float), cudaMemcpyHostToDevice);

                countElements<float><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_f, d_output_f, input.row_count);
                cudaDeviceSynchronize();

                float count_value;
                cudaMemcpy(&count_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                result.data[i] = new float(count_value);

                cudaFree(d_input_f);
                cudaFree(d_output_f);
                break;
            }
            case DataType::DATETIME:
            {
                uint64_t *d_input_ui = nullptr;
                float *d_output_f = nullptr;
                cudaMalloc(&d_input_ui, input.row_count * sizeof(uint64_t));
                cudaMalloc(&d_output_f, sizeof(float));
                cudaMemset(d_output_f, 0, sizeof(float)); // Initialize output to 0

                cudaMemcpy(d_input_ui, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

                countElements<uint64_t><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_ui, d_output_f, input.row_count);
                cudaDeviceSynchronize();

                float count_value;
                cudaMemcpy(&count_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                result.data[i] = new float(count_value);

                cudaFree(d_input_ui);
                cudaFree(d_output_f);
                break;
            }
            case DataType::STRING:
            {
                // Allocate device memory for string pointers and output
                char **d_input = nullptr;
                float *d_output = nullptr;
                cudaMalloc(&d_input, input.row_count * sizeof(char *));
                cudaMalloc(&d_output, sizeof(float));
                cudaMemset(d_output, 0, sizeof(float));

                // First copy the host pointers to a temporary array
                char **host_pointers = new char *[input.row_count];
                const char **input_column = static_cast<const char **>(input.data[aggregates[i].column_index]);

                // Allocate device memory for each string and copy its content
                for (size_t j = 0; j < input.row_count; j++)
                {
                    if (input_column[j] != nullptr)
                    {
                        size_t len = strlen(input_column[j]) + 1;
                        cudaMalloc(&host_pointers[j], len);
                        cudaMemcpy(host_pointers[j], input_column[j], len, cudaMemcpyHostToDevice);
                    }
                    else
                    {
                        host_pointers[j] = nullptr;
                    }
                }

                // Copy the pointer array to device
                cudaMemcpy(d_input, host_pointers, input.row_count * sizeof(char *), cudaMemcpyHostToDevice);

                // Launch kernel
                countElements<char *><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input, d_output, input.row_count);

                // Get result
                float count_value;
                cudaMemcpy(&count_value, d_output, sizeof(float), cudaMemcpyDeviceToHost);
                result.data[i] = new float(count_value);

                // Cleanup
                for (size_t j = 0; j < input.row_count; j++)
                {
                    if (host_pointers[j] != nullptr)
                    {
                        cudaFree(host_pointers[j]);
                    }
                }
                delete[] host_pointers;
                cudaFree(d_input);
                cudaFree(d_output);
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for Count aggregate");
            }
        }
        break;
            /*SUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUMSUM*/
        case AggregateType::SUM:
            if (input.columns[aggregates[i].column_index].type == DataType::FLOAT)
            {
                float *h_input = static_cast<float *>(input.data[aggregates[i].column_index]);

                float *d_input_f = nullptr, *d_output_f = nullptr;
                cudaMalloc(&d_input_f, input.row_count * sizeof(float));
                cudaMalloc(&d_output_f, sizeof(float));

                cudaMemcpy(d_input_f, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(float), cudaMemcpyHostToDevice);

                findSumElement<<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_f, d_output_f, input.row_count);

                cudaDeviceSynchronize();
                float f_value;

                cudaMemcpy(&f_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "sum value: " << f_value << std::endl;
                result.data[i] = new float(f_value);

                cudaFree(d_input_f);
                cudaFree(d_output_f);
            }
            break;
            /*AVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVGAVG*/
        case AggregateType::AVG:
            if (input.columns[aggregates[i].column_index].type == DataType::FLOAT)
            {
                float *h_input = static_cast<float *>(input.data[aggregates[i].column_index]);

                float *d_input_f = nullptr, *d_output_f = nullptr;
                cudaMalloc(&d_input_f, input.row_count * sizeof(float));
                cudaMalloc(&d_output_f, sizeof(float));

                cudaMemcpy(d_input_f, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(float), cudaMemcpyHostToDevice);

                findSumElement<<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_f, d_output_f, input.row_count);

                cudaDeviceSynchronize();
                float f_value;

                cudaMemcpy(&f_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                f_value /= input.row_count;
                std::cout << "AVG value: " << f_value << std::endl;
                result.data[i] = new float(f_value);

                cudaFree(d_input_f);
                cudaFree(d_output_f);
            }
            break;
            /*MINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMINMIN*/
        case AggregateType::MIN:
            switch (input.columns[aggregates[i].column_index].type)
            {
            case DataType::FLOAT:
            {
                float *h_input = static_cast<float *>(input.data[aggregates[i].column_index]);

                float *d_input_f = nullptr, *d_output_f = nullptr;
                cudaMalloc(&d_input_f, input.row_count * sizeof(float));
                cudaMalloc(&d_output_f, sizeof(float));

                float init_val = FLT_MAX;
                cudaMemcpy(d_output_f, &init_val, sizeof(float), cudaMemcpyHostToDevice);

                cudaMemcpy(d_input_f, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(float), cudaMemcpyHostToDevice);

                findMinElement<float><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_f, d_output_f, input.row_count);

                cudaDeviceSynchronize();
                float f_value;

                cudaMemcpy(&f_value, d_output_f, sizeof(float), cudaMemcpyDeviceToHost);
                std::cout << "Min value: " << f_value << std::endl;
                result.data[i] = new float(f_value);

                cudaFree(d_input_f);
                cudaFree(d_output_f);
                break;
            }
            case DataType::DATETIME:
            {

                uint64_t *d_input_ui = nullptr, *d_output_ui = nullptr;
                cudaMalloc(&d_input_ui, input.row_count * sizeof(uint64_t));
                cudaMalloc(&d_output_ui, sizeof(uint64_t));

                uint64_t init_val = UINT64_MAX;
                std::cout << "init_val: " << init_val << std::endl;
                cudaMemcpy(d_output_ui, &init_val, sizeof(float), cudaMemcpyHostToDevice);

                cudaMemcpy(d_input_ui, input.data[aggregates[i].column_index],
                           input.row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);

                findMinElement<uint64_t><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_ui, d_output_ui, input.row_count);

                uint64_t datetime_value;
                cudaMemcpy(&datetime_value, d_output_ui, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                std::cout << "MIN datetime value: " << datetime_value << std::endl;
                result.data[i] = new uint64_t(datetime_value);

                cudaFree(d_input_ui);
                cudaFree(d_output_ui);
                break;
            }
            case DataType::STRING:
            {
                char **d_input_char = nullptr, **d_output_char = nullptr;
                cudaMalloc(&d_input_char, input.row_count * sizeof(char *));
                cudaMalloc(&d_output_char, sizeof(char *));

                char **d_strings = new char *[input.row_count];
                const char **host_strings = static_cast<const char **>(input.data[aggregates[i].column_index]);

                for (size_t j = 0; j < input.row_count; j++)
                {
                    size_t len = strlen(host_strings[j]) + 1;
                    cudaMalloc(&d_strings[j], len);
                    cudaMemcpy(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice);
                    cudaMemcpy(&d_input_char[j], &d_strings[j], sizeof(char *), cudaMemcpyHostToDevice);
                }

                findMinElement<char *><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_char, d_output_char, input.row_count);
                cudaDeviceSynchronize();

                char *d_max_string = nullptr;
                cudaMemcpy(&d_max_string, d_output_char, sizeof(char *), cudaMemcpyDeviceToHost);

                size_t max_len = 0;
                if (d_max_string)
                {
                    char temp_char;
                    size_t offset = 0;
                    do
                    {
                        cudaMemcpy(&temp_char, d_max_string + offset, 1, cudaMemcpyDeviceToHost);
                        offset++;
                    } while (temp_char != '\0');
                    max_len = offset;
                }

                // Allocate host memory for the result string
                char *h_output_string = new char[max_len];
                cudaMemcpy(h_output_string, d_max_string, max_len, cudaMemcpyDeviceToHost);
                // std::cout << "Max string: " << h_output_string << std::endl;

                result.data[i] = new char *[1];
                static_cast<char **>(result.data[i])[0] = h_output_string;
                std::cout << "MIN string: " << static_cast<char **>(result.data[i])[0] << std::endl;
                // result.print();
                // Cleanup
                for (size_t j = 0; j < input.row_count; j++)
                {
                    cudaFree(d_strings[j]);
                }
                delete[] d_strings;
                cudaFree(d_input_char);
                cudaFree(d_output_char);
                break;
            }
            default:
                throw std::runtime_error("Unsupported data type for MAX aggregate");
            }
            break;
        /* MAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAXMAX*/
        case AggregateType::MAX:
            switch (input.columns[aggregates[i].column_index].type)
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
                char **d_input_char = nullptr, **d_output_char = nullptr;
                cudaMalloc(&d_input_char, input.row_count * sizeof(char *));
                cudaMalloc(&d_output_char, sizeof(char *));

                char **d_strings = new char *[input.row_count];
                const char **host_strings = static_cast<const char **>(input.data[aggregates[i].column_index]);

                for (size_t j = 0; j < input.row_count; j++)
                {
                    size_t len = strlen(host_strings[j]) + 1;
                    cudaMalloc(&d_strings[j], len);
                    cudaMemcpy(d_strings[j], host_strings[j], len, cudaMemcpyHostToDevice);
                    cudaMemcpy(&d_input_char[j], &d_strings[j], sizeof(char *), cudaMemcpyHostToDevice);
                }

                findMaxElement<char *><<<numBlocks, numThreads, shared_mem_size>>>(
                    d_input_char, d_output_char, input.row_count);
                cudaDeviceSynchronize();

                char *d_max_string = nullptr;
                cudaMemcpy(&d_max_string, d_output_char, sizeof(char *), cudaMemcpyDeviceToHost);

                size_t max_len = 0;
                if (d_max_string)
                {
                    char temp_char;
                    size_t offset = 0;
                    do
                    {
                        cudaMemcpy(&temp_char, d_max_string + offset, 1, cudaMemcpyDeviceToHost);
                        offset++;
                    } while (temp_char != '\0');
                    max_len = offset;
                }

                // Allocate host memory for the result string
                char *h_output_string = new char[max_len];
                cudaMemcpy(h_output_string, d_max_string, max_len, cudaMemcpyDeviceToHost);
                // std::cout << "Max string: " << h_output_string << std::endl;

                result.data[i] = new char *[1];
                static_cast<char **>(result.data[i])[0] = h_output_string;
                std::cout << "Max string: " << static_cast<char **>(result.data[i])[0] << std::endl;
                // result.print();
                // Cleanup
                for (size_t j = 0; j < input.row_count; j++)
                {
                    cudaFree(d_strings[j]);
                }
                delete[] d_strings;
                cudaFree(d_input_char);
                cudaFree(d_output_char);
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
