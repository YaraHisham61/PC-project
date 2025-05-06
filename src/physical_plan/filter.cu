#include "physical_plan/filter.hpp"

Filter::Filter(const duckdb::InsertionOrderPreservingMap<std::string> &params)
    : PhysicalOpNode()
{
    auto it = params.find("__expression__");
    if (it != params.end())
    {
        parseConditions(it->second);
        removeTimestampSuffixSimple();
    }
}
void Filter::removeTimestampSuffixSimple()
{
    const std::string suffix = "::TIMESTAMP";

    for (auto &cond : conditions)
    {
        size_t pos = cond.value.rfind(suffix);
        if (pos != std::string::npos)
        {
            cond.value = cond.value.substr(1, pos - 1);
        }
    }
}
std::string Filter::trim(const std::string &str) const
{
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
        return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

void Filter::parseConditions(const std::string &expression)
{
    std::string expr = trim(expression);

    size_t and_pos = expr.find(" AND ");
    size_t or_pos = expr.find(" OR ");

    if (and_pos != std::string::npos || or_pos != std::string::npos)
    {
        parseComplexCondition(expr);
    }
    else
    {
        parseSimpleCondition(expr);
    }
}

void Filter::parseSimpleCondition(const std::string &expr)
{
    std::string inner = trim(expr);
    if (inner.front() == '(' && inner.back() == ')')
    {
        inner = inner.substr(1, inner.length() - 2);
    }

    Condition cond = parseSingleCondition(inner);
    if (!cond.column.empty())
    {
        conditions.push_back(cond);
    }
}

void Filter::parseComplexCondition(const std::string &expr)
{
    std::string inner = trim(expr);
    if (inner.front() == '(' && inner.back() == ')')
    {
        inner = inner.substr(1, inner.length() - 2);
    }

    std::vector<std::string> tokens;
    std::string current;
    int paren_level = 0;

    for (char c : inner)
    {
        if (c == '(')
            paren_level++;
        if (c == ')')
            paren_level--;

        if (paren_level == 0 && c == ' ')
        {
            if (!current.empty())
            {
                tokens.push_back(current);
                current.clear();
            }
        }
        else
        {
            current += c;
        }
    }
    if (!current.empty())
        tokens.push_back(current);

    for (size_t i = 0; i < tokens.size();)
    {
        if (tokens[i] == "AND" || tokens[i] == "OR")
        {
            logical_ops.push_back(tokens[i]);
            i++;
        }
        else
        {
            parseSimpleCondition(tokens[i]);
            i++;
        }
    }
}

Condition Filter::parseSingleCondition(const std::string &cond_expr) const
{
    Condition cond;
    std::regex op_regex(R"((>=|<=|!=|>|<|=))");
    std::smatch op_match;

    if (std::regex_search(cond_expr, op_match, op_regex))
    {
        cond.column = trim(cond_expr.substr(0, op_match.position()));
        cond.op = op_match.str();
        cond.value = trim(cond_expr.substr(op_match.position() + cond.op.length()));

        if (!cond.value.empty() && cond.value.front() == '\'' && cond.value.back() == '\'')
        {
            cond.value = cond.value.substr(1, cond.value.length() - 2);
            cond.is_string = true;
        }
        else
        {
            cond.is_string = false;
        }
    }
    return cond;
}

bool *Filter::getSelectedRows(const TableResults &input_table) const
{
    const size_t row_count = input_table.row_count;
    bool *h_final_mask = new bool[row_count]();

    bool *d_combined_mask = nullptr;
    cudaMalloc(&d_combined_mask, row_count * sizeof(bool));
    cudaMemset(d_combined_mask, 0, row_count * sizeof(bool));

    int numThreads = 256;
    int numBlocks = (row_count + numThreads - 1) / numThreads;

    for (size_t cond_idx = 0; cond_idx < this->conditions.size(); cond_idx++)
    {
        const auto &cond = this->conditions[cond_idx];
        const size_t col_idx = input_table.getColumnIndex(cond.column);
        const DataType col_type = input_table.columns[col_idx].type;

        uint8_t cond_code = 0;
        if (cond.op == ">")
            cond_code = 1;
        else if (cond.op == "<")
            cond_code = 2;
        else if (cond.op == "=")
            cond_code = 3;
        else if (cond.op == "!=")
            cond_code = 4;
        else if (cond.op == "<=")
            cond_code = 5;
        else if (cond.op == ">=")
            cond_code = 6;
        else
        {
            std::cerr << "Unsupported operator: " << cond.op << "\n";
            continue;
        }

        bool *d_temp_mask = nullptr;
        cudaMalloc(&d_temp_mask, row_count * sizeof(bool));
        cudaMemset(d_temp_mask, 0, row_count * sizeof(bool));

        if (col_type == DataType::FLOAT)
        {
            float *d_col_data = nullptr;
            cudaMalloc(&d_col_data, row_count * sizeof(float));
            cudaMemcpy(d_col_data, input_table.data[col_idx], row_count * sizeof(float), cudaMemcpyHostToDevice);
            float value = std::stof(cond.value);
            filterKernel<float><<<numBlocks, numThreads>>>(d_col_data, d_temp_mask, row_count, value, cond_code);
            cudaFree(d_col_data);
        }
        else if (col_type == DataType::DATETIME)
        {
            uint64_t *d_col_data = nullptr;
            cudaMalloc(&d_col_data, row_count * sizeof(uint64_t));
            cudaMemcpy(d_col_data, input_table.data[col_idx], row_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
            uint64_t value = getDateTime(cond.value);
            filterKernel<uint64_t><<<numBlocks, numThreads>>>(d_col_data, d_temp_mask, row_count, value, cond_code);
            cudaFree(d_col_data);
        }
        else if (col_type == DataType::STRING)
        {
            const char **d_col_data = nullptr;
            cudaMalloc(&d_col_data, row_count * sizeof(char *));
            const char **host_strings = static_cast<const char **>(input_table.data[col_idx]);

            char **d_strings = new char *[row_count];

            for (size_t i = 0; i < row_count; i++)
            {
                size_t len = strlen(host_strings[i]) + 1;
                cudaMalloc(&d_strings[i], len);
                cudaMemcpy(d_strings[i], host_strings[i], len, cudaMemcpyHostToDevice);
                cudaMemcpy(&d_col_data[i], &d_strings[i], sizeof(char *), cudaMemcpyHostToDevice);
            }

            char *d_value = nullptr;
            cudaMalloc(&d_value, cond.value.size() + 1);
            cudaMemcpy(d_value, cond.value.c_str(), cond.value.size() + 1, cudaMemcpyHostToDevice);
            filterKernelString<<<numBlocks, numThreads>>>(d_col_data, d_temp_mask, row_count, d_value, cond_code);

            for (size_t i = 0; i < row_count; i++)
            {
                cudaFree(d_strings[i]);
            }
            delete[] d_strings;
            cudaFree(d_value);
            cudaFree(d_col_data);
        }

        if (cond_idx == 0)
        {
            cudaMemcpy(d_combined_mask, d_temp_mask, row_count * sizeof(bool), cudaMemcpyDeviceToDevice);
        }
        else
        {
            const std::string &op = logical_ops[cond_idx - 1];
            if (op == "AND")
            {
                andKernel<<<numBlocks, numThreads>>>(d_combined_mask, d_temp_mask, row_count);
            }
            else if (op == "OR")
            {
                orKernel<<<numBlocks, numThreads>>>(d_combined_mask, d_temp_mask, row_count);
            }
        }
        bool *h_temp_mask = new bool[row_count];
        cudaMemcpy(h_temp_mask, d_temp_mask, row_count * sizeof(bool), cudaMemcpyDeviceToHost);

        cudaFree(d_temp_mask);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_final_mask, d_combined_mask, row_count * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_combined_mask);

    size_t filtered_row_count = 0;
    for (size_t i = 0; i < row_count; i++)
    {
        if (h_final_mask[i])
            filtered_row_count++;
    }

    // std::cout << "Total rows matched: " << filtered_row_count << "\n";
    return h_final_mask;
}

TableResults Filter::applyFilter(const TableResults &input_table) const
{
    if (input_table.row_count == 0)
    {
        return input_table;
    }
    std::unique_ptr<bool[], void (*)(bool *)> h_selected_rows(
        getSelectedRows(input_table),
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

    // Create streams: one for prefix sum and one per column
    cudaStream_t prefix_stream;
    cudaStreamCreate(&prefix_stream);
    std::vector<cudaStream_t> col_streams(input_table.column_count);
    for (auto &stream : col_streams)
    {
        cudaStreamCreate(&stream);
    }

    bool *d_mask = nullptr;
    unsigned int *d_positions;

    cudaMalloc(&d_mask, row_count * sizeof(bool));
    cudaMalloc(&d_positions, row_count * sizeof(unsigned int));

    cudaMemcpyAsync(d_mask, h_selected_rows.get(), row_count * sizeof(bool), cudaMemcpyHostToDevice, prefix_stream);

    int threads = 256;
    int blocks = (row_count + threads - 1) / threads;
    size_t shared_mem = threads * sizeof(unsigned int);

    computeOutputPositions<<<blocks, threads, shared_mem, prefix_stream>>>(d_mask, d_positions, row_count);
    cudaStreamSynchronize(prefix_stream);

    for (size_t col_idx = 0; col_idx < input_table.column_count; col_idx++)
    {
        const DataType col_type = input_table.columns[col_idx].type;

        switch (col_type)
        {
        case DataType::FLOAT:
        {
            float *h_input_data = static_cast<float *>(input_table.data[col_idx]);
            float *d_input = nullptr;
            float *d_output = nullptr;
            cudaMalloc(&d_input, row_count * sizeof(float));
            cudaMalloc(&d_output, selected_count * sizeof(float));
            std::unique_ptr<float[], void (*)(float *)> h_output_data(
                static_cast<float *>(malloc(selected_count * sizeof(float))),
                [](float *ptr)
                { free(ptr); });

            cudaMemcpyAsync(d_input, h_input_data, row_count * sizeof(float), cudaMemcpyHostToDevice, col_streams[col_idx]);
            copySelectedRowsKernel<float><<<blocks, threads, 0, col_streams[col_idx]>>>(d_input, d_output, d_mask, d_positions, row_count);
            cudaMemcpyAsync(h_output_data.get(), d_output, selected_count * sizeof(float), cudaMemcpyDeviceToHost, col_streams[col_idx]);

            filtered_table.data[col_idx] = h_output_data.release();
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *h_input_data = static_cast<uint64_t *>(input_table.data[col_idx]);
            uint64_t *d_input = nullptr;
            uint64_t *d_output = nullptr;
            cudaMalloc(&d_input, row_count * sizeof(uint64_t));
            cudaMalloc(&d_output, selected_count * sizeof(uint64_t));
            std::unique_ptr<uint64_t[], void (*)(uint64_t *)> h_output_data(
                static_cast<uint64_t *>(malloc(selected_count * sizeof(uint64_t))),
                [](uint64_t *ptr)
                { free(ptr); });

            cudaMemcpyAsync(d_input, h_input_data, row_count * sizeof(uint64_t), cudaMemcpyHostToDevice, col_streams[col_idx]);
            copySelectedRowsKernel<uint64_t><<<blocks, threads, 0, col_streams[col_idx]>>>(d_input, d_output, d_mask, d_positions, row_count);
            cudaMemcpyAsync(h_output_data.get(), d_output, selected_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, col_streams[col_idx]);

            filtered_table.data[col_idx] = h_output_data.release();
            cudaFree(d_input);
            cudaFree(d_output);
            break;
        }
        case DataType::STRING:
        {
            const char **h_input_strings = static_cast<const char **>(input_table.data[col_idx]);
            std::unique_ptr<const char *[], void (*)(const char **)>
                h_output_strings(
                    static_cast<const char **>(malloc(selected_count * sizeof(char *))),
                    [](const char **ptr)
                    { free(ptr); });

            const char **d_input_strings = nullptr;
            cudaMalloc(&d_input_strings, row_count * sizeof(char *));
            cudaMemcpyAsync(d_input_strings, h_input_strings, row_count * sizeof(char *), cudaMemcpyHostToDevice, col_streams[col_idx]);

            const char **d_output_strings = nullptr;
            cudaMalloc(&d_output_strings, selected_count * sizeof(char *));

            copySelectedStringRowsKernel<<<blocks, threads, 0, col_streams[col_idx]>>>(d_input_strings, d_output_strings, d_mask, d_positions, row_count);
            cudaMemcpyAsync(h_output_strings.get(), d_output_strings, selected_count * sizeof(char *), cudaMemcpyDeviceToHost, col_streams[col_idx]);

            filtered_table.data[col_idx] = h_output_strings.release();
            cudaFree(d_input_strings);
            cudaFree(d_output_strings);
            break;
        }
        default:
            cudaStreamDestroy(prefix_stream);
            for (auto &stream : col_streams)
                cudaStreamDestroy(stream);
            cudaFree(d_mask);
            cudaFree(d_positions);
            throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int>(col_type)));
        }
    }

    for (auto &stream : col_streams)
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    cudaStreamDestroy(prefix_stream);
    cudaFree(d_mask);
    cudaFree(d_positions);

    return filtered_table;
}
void Filter::print() const
{
    std::cout << "FILTER (";
    for (size_t i = 0; i < conditions.size(); ++i)
    {
        if (i > 0 && i - 1 < logical_ops.size())
        {
            std::cout << " " << logical_ops[i - 1] << " ";
        }
        std::cout << conditions[i].column << " " << conditions[i].op << " ";
        if (conditions[i].is_string)
        {
            std::cout << "'" << conditions[i].value << "'";
        }
        else
        {
            std::cout << conditions[i].value;
        }
    }
    std::cout << ")\n";
}