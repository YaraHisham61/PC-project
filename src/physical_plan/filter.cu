#include "physical_plan/filter.hpp"

Filter::Filter(const duckdb::InsertionOrderPreservingMap<std::string> &params)
    : PhysicalOpNode()
{
    auto it = params.find("__expression__");
    if (it != params.end())
    {
        parseConditions(it->second);
    }
    std::cout << "Filter: " << conditions.size() << " conditions\n";
    std::cout << "Logical ops: ";
    for (const auto &op : logical_ops)
    {
        std::cout << op << " ";
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

TableResults Filter::applyFilter(const TableResults &input_table) const
{
    TableResults filtered_table;
    filtered_table.columns = input_table.columns;
    filtered_table.column_count = input_table.column_count;

    size_t num_cols = input_table.column_count;

    for (const auto &cond : this->conditions)
    {

        ColumnInfo col;
        col.name = cond.column;
        col.idx = input_table.getColumnIndex(cond.column);
        col.type = input_table.columns[col.idx].type;
        void *d_col_in;
        bool *d_col_out_bool;
        cudaMalloc(&d_col_out_bool, input_table.row_count * sizeof(bool *));

        if (col.type == DataType::INT)
        {
            d_col_in = static_cast<int *>(d_col_in);
            cudaMalloc(&d_col_in, input_table.row_count * sizeof(int));
            cudaMemcpy(d_col_in, input_table.data[col.idx], input_table.row_count * sizeof(int), cudaMemcpyHostToDevice);
        }
        else if (col.type == DataType::FLOAT)
        {
            d_col_in = static_cast<float *>(d_col_in);
            cudaMalloc(&d_col_in, input_table.row_count * sizeof(float));
            cudaMemcpy(d_col_in, input_table.data[col.idx], input_table.row_count * sizeof(float), cudaMemcpyHostToDevice);
        }
        else if (col.type == DataType::DATETIME)
        {
            d_col_in = static_cast<int64_t *>(d_col_in);
            cudaMalloc(&d_col_in, input_table.row_count * sizeof(int64_t));
            cudaMemcpy(d_col_in, input_table.data[col.idx], input_table.row_count * sizeof(int64_t), cudaMemcpyHostToDevice);
        }
        else if (col.type == DataType::STRING)
        {
            d_col_in = static_cast<char **>(d_col_in);
            cudaMalloc(&d_col_in, input_table.row_count * sizeof(char *));
            cudaMemcpy(d_col_in, input_table.data[col.idx], input_table.row_count * sizeof(char *), cudaMemcpyHostToDevice);
        }

        uint8_t cond_int = 0;
        if (cond.op == ">")
        {
            cond_int = 1;
        }
        else if (cond.op == "<")
        {
            cond_int = 2;
        }
        else if (cond.op == "==")
        {
            cond_int = 3;
        }
        else if (cond.op == "!=")
        {
            cond_int = 4;
        }
        else if (cond.op == "<=")
        {
            cond_int = 5;
        }
        else if (cond.op == ">=")
        {
            cond_int = 6;
        }
        else
        {
            std::cerr << "Unsupported operator: " << cond.op << "\n";
        }
        int numThreads = 1024;
        float value = 2019;
        int numBlocks = (input_table.row_count + numThreads - 1) / numThreads;
        filterKernel<<<numBlocks, numThreads>>>(d_col_in, d_col_out_bool, input_table.row_count, value, cond_int);
        cudaDeviceSynchronize();

        bool *h_col_out_bool;
        h_col_out_bool = (bool *)malloc(input_table.row_count * sizeof(bool));
        cudaMemcpy(h_col_out_bool, d_col_out_bool, input_table.row_count * sizeof(bool), cudaMemcpyDeviceToHost);
        for (int i = 0; i < input_table.row_count; ++i)
        {
            std::cout << h_col_out_bool[i] << " ";
        }
    }

    // {
    //     filtered_table.columns.push_back(col);
    // }
    // for (size_t i = 0; i < input_table.row_count; ++i)
    // {
    //     bool row_matches = evaluateRow(input_table, i, num_cols);

    //     if (row_matches)
    //     {
    //         for (size_t c = 0; c < num_cols; ++c)
    //         {
    //             filtered_table.rows.push_back(input_table.rows[i * num_cols + c]);
    //         }
    //         filtered_table.row_count++;
    //     }
    // }

    return filtered_table;
}

// bool Filter::evaluateRow(const TableResults &table, size_t row_idx, size_t num_cols) const
// {
//     if (conditions.empty())
//         return true;

//     bool result = evaluateCondition(table, row_idx, num_cols, conditions[0]);
//     size_t op_index = 0;

//     for (size_t i = 1; i < conditions.size(); i++)
//     {
//         bool current = evaluateCondition(table, row_idx, num_cols, conditions[i]);

//         if (op_index < logical_ops.size())
//         {
//             if (logical_ops[op_index] == "AND")
//             {
//                 result = result && current;
//             }
//             else if (logical_ops[op_index] == "OR")
//             {
//                 result = result || current;
//             }
//             op_index++;
//         }
//         else
//         {
//             result = result && current;
//         }
//     }

//     return result;
// }

// bool Filter::evaluateCondition(const TableResults &table, size_t row_idx, size_t num_cols, const Condition &cond) const
// {
//     int col_idx = -1;
//     for (size_t c = 0; c < table.columns.size(); ++c)
//     {
//         if (table.columns[c].name == cond.column)
//         {
//             col_idx = c;
//             break;
//         }
//     }
//     if (col_idx == -1)
//     {
//         std::cerr << "Column " << cond.column << " not found!\n";
//         return false;
//     }

//     const auto &cell = table.rows[row_idx * num_cols + col_idx];

//     if (cond.is_string)
//     {
//         if (!std::holds_alternative<std::string>(cell))
//         {
//             return false;
//         }
//         const std::string &cell_str = std::get<std::string>(cell);

//         if (cond.op == "=")
//             return cell_str == cond.value;
//         if (cond.op == "!=")
//             return cell_str != cond.value;

//         std::cerr << "Operator " << cond.op << " not supported for string columns.\n";
//         return false;
//     }
//     else
//     {
//         double cell_num = 0.0;
//         if (std::holds_alternative<int>(cell))
//         {
//             cell_num = static_cast<double>(std::get<int>(cell));
//         }
//         else if (std::holds_alternative<int64_t>(cell))
//         {
//             cell_num = static_cast<double>(std::get<int64_t>(cell));
//         }
//         else if (std::holds_alternative<float>(cell))
//         {
//             cell_num = static_cast<double>(std::get<float>(cell));
//         }

//         double value_num = std::stod(cond.value);

//         if (cond.op == ">")
//             return cell_num > value_num;
//         if (cond.op == "<")
//             return cell_num < value_num;
//         if (cond.op == ">=")
//             return cell_num >= value_num;
//         if (cond.op == "<=")
//             return cell_num <= value_num;
//         if (cond.op == "=")
//             return std::abs(cell_num - value_num) < 1e-9;
//         if (cond.op == "!=")
//             return std::abs(cell_num - value_num) >= 1e-9;

//         std::cerr << "Operator " << cond.op << " not supported.\n";
//         return false;
//     }
// }

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