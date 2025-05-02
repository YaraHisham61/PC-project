#include "physical_plan/projection.hpp"

Projection::Projection(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("__projections__");
    if (it != params.end())
    {
        parseProjectionList(it->second);
    }
}

void Projection::parseProjectionList(const std::string &projection_list)
{
    std::istringstream iss(projection_list);
    std::string line;

    flag = projection_list.find('#') != std::string::npos;

    while (std::getline(iss, line))
    {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty())
            continue;
        if (flag)
        {
            projections_index.push_back(line[1] - '0');
        }
        else
        {
            output_names.push_back(line);
        }
    }
}

TableResults Projection::applyProjection(const TableResults &input_table) const
{
    TableResults result;
    auto modified_names = this->output_names;

    if (input_table.column_count == 0 || input_table.row_count == 0)
    {
        return result;
    }

    if (flag)
    {
        for (int index : projections_index)
        {
            if (index < 0 || index >= static_cast<int>(input_table.column_count))
            {
                throw std::runtime_error("Projection index out of bounds");
            }
            result.columns.push_back(input_table.columns[index]);
        }

        result.column_count = projections_index.size();
        result.row_count = input_table.row_count;
        result.data.resize(result.column_count);

        for (size_t col_idx = 0; col_idx < result.columns.size(); ++col_idx)
        {
            const ColumnInfo &col_info = result.columns[col_idx];
            int input_col_idx = projections_index[col_idx];

            switch (col_info.type)
            {
            case DataType::FLOAT:
                result.data[col_idx] = static_cast<float *>(malloc(result.row_count * sizeof(float)));
                memcpy(result.data[col_idx], input_table.data[input_col_idx], result.row_count * sizeof(float));
                break;
            case DataType::DATETIME:
                result.data[col_idx] = static_cast<int64_t *>(malloc(result.row_count * sizeof(int64_t)));
                memcpy(result.data[col_idx], input_table.data[input_col_idx], result.row_count * sizeof(int64_t));
                break;
            case DataType::STRING:
                result.data[col_idx] = static_cast<char **>(malloc(result.row_count * sizeof(char *)));
                for (size_t row = 0; row < result.row_count; ++row)
                {
                    char *original_str = static_cast<char **>(input_table.data[input_col_idx])[row];
                    static_cast<char **>(result.data[col_idx])[row] = strdup(original_str);
                }
                break;
            default:
                break;
            }
        }
    }
    else
    {
        result.column_count = modified_names.size();
        result.row_count = input_table.row_count;
        result.data.resize(result.column_count);
        // int col_idx = 0;
        for (size_t col_idx = 0; col_idx < modified_names.size(); ++col_idx)
        {

            auto &name = modified_names[col_idx];
            if (name.find("CAST(") != std::string::npos)
            {
                modified_names[col_idx] = extract_base_column_name(name);
            }

            ColumnInfo col_info;
            size_t input_index = input_table.getColumnIndex(name);
            col_info = input_table.columns[input_index];
            col_info.idx = col_idx;
            result.columns.push_back(col_info);
            switch (col_info.type)
            {
            case DataType::FLOAT:
                result.data[col_idx] = static_cast<float *>(malloc(result.row_count * sizeof(float)));
                result.data[col_idx] = input_table.data[input_index];
                break;

            case DataType::DATETIME:
                result.data[col_idx] = static_cast<int64_t *>(malloc(result.row_count * sizeof(int64_t)));
                result.data[col_idx] = input_table.data[input_index];
                break;
            case DataType::STRING:
                result.data[col_idx] = static_cast<char **>(malloc(result.row_count * sizeof(char *)));
                result.data[col_idx] = input_table.data[input_index];
                break;
            default:
                break;
            }
        }
    }

    if (!modified_names.empty())
    {
        if (modified_names.size() != result.columns.size())
        {
            
            throw std::runtime_error("Output names count doesn't match projection result columns count");
        }

        for (size_t i = 0; i < result.columns.size(); ++i)
        {

            result.columns[i].name = modified_names[i];
        }
    }

    return result;
}
void Projection::print() const
{
    std::cout << "Projection: ";
    for (const auto &name : output_names)
    {
        std::cout << name << " ";
    }
    std::cout << "\n";
}

std::string Projection::extract_base_column_name(std::string column_name) const
{
    if (column_name.find("CAST(") == 0 && column_name.find(")") != std::string::npos)
    {
        size_t start = column_name.find("(") + 1;
        size_t end = column_name.find(" AS ");

        if (end != std::string::npos)
        {
            return column_name.substr(start, end - start);
        }
    }
    return column_name;
}