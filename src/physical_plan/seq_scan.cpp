#include "physical_plan/seq_scan.hpp"

SeqScan::SeqScan(const duckdb::InsertionOrderPreservingMap<std::string> &params)
    : PhysicalOpNode()
{
    for (const auto &p : params)
    {
        if (p.first == "__estimated_cardinality__")
            continue;
        if (p.first == "Table")
        {
            this->table_name = p.second;
        }
        if (p.first == "Projections")
        {
            std::stringstream ss(p.second);
            std::string col;
            while (std::getline(ss, col))
            {
                col.erase(0, col.find_first_not_of(" \t\r\n"));
                col.erase(col.find_last_not_of(" \t\r\n") + 1);
                if (!col.empty())
                    this->projections.push_back(col);
            }
        }
    }
}

void SeqScan::print() const
{
    std::cout << "SEQ_SCAN Node:\n";
    std::cout << "  Table: " << this->table_name << "\n";
    std::cout << "  Projections: ";
    for (const auto &col : projections)
    {
        std::cout << col << " ";
    }
    std::cout << "\n";
    for (const auto &child : children)
    {
        child->print(); // Polymorphic call to child’s print
    }
}

TableResults SeqScan::read_scan_table(DB *data_base)
{
    TableResults result;
    std::string csv_file = DATA_DIR + this->table_name + ".csv";
    rapidcsv::Document doc(csv_file);
    std::vector<std::string> cols_to_read = this->projections;

    if (cols_to_read.empty())
    {
        cols_to_read = data_base->get_table(table_name)->getColumnNames();
    }
    int idx = 0;
    for (const auto &col_name : cols_to_read)
    {
        ColumnInfo *c = data_base->get_column(table_name, col_name);
        c->idx = idx++;
        if (!c)
        {
            std::cerr << "Column " << col_name << " not found in table " << table_name << std::endl;
            continue;
        }
        result.columns.push_back(*c);
    }
    result.column_count = result.columns.size();
    result.row_count = doc.GetRowCount();
    result.data.resize(result.column_count);    

    for (const auto &col : result.columns)
    {
        std::string col_name = get_original_column_name(&col, cols_to_read[col.idx]);

        switch (col.type)
        {
        case DataType::FLOAT:
            result.data[col.idx] = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                float value = doc.GetCell<float>(col_name, row_idx);
                static_cast<float *>(result.data[col.idx])[row_idx] = value;
            }
            break;
        case DataType::INT:
            result.data[col.idx] = static_cast<int *>(malloc(result.row_count * sizeof(int)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                static_cast<int *>(result.data[col.idx])[row_idx] = doc.GetCell<int>(col_name, row_idx);
            }
            break;
        case DataType::DATETIME:
            result.data[col.idx] = static_cast<int64_t *>(malloc(result.row_count * sizeof(int64_t)));

            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                static_cast<int64_t *>(result.data[col.idx])[row_idx] = doc.GetCell<int64_t>(col_name, row_idx);
            }
            break;
        case DataType::STRING:
            result.data[col.idx] = static_cast<char **>(malloc(result.row_count * sizeof(char *)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                std::string str_value = doc.GetCell<std::string>(col_name, row_idx);
                static_cast<char **>(result.data[col.idx])[row_idx] = strdup(str_value.c_str());
            }
            break;
        default:
            break;
        }
    }

    return result;
}

std::string SeqScan::get_original_column_name(const ColumnInfo *c, const std::string &col_name)
{
    std::string new_col_name = col_name;

    if (c->type == DataType::STRING)
    {
        new_col_name = col_name + " (T)";
    }
    else if (c->type == DataType::DATETIME)
    {
        new_col_name = col_name + " (D)";
    }
    else if (c->type == DataType::FLOAT || c->type == DataType::INT)
    {
        new_col_name = col_name + " (N)";
    }

    if (c->is_primary)
    {
        new_col_name += " (P)";
    }
    return new_col_name;
}
