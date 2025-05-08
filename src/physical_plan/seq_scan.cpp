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

TableResults SeqScan::read_scan_table(DB *data_base, size_t batch_index, size_t batch_size)
{
    TableResults result;
    result.batch_index = batch_index;

    std::string csv_file = DATA_DIR + this->table_name + ".csv";

    // Open file stream
    std::ifstream file(csv_file);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << csv_file << std::endl;
        result.row_count = 0;
        result.has_more = false;
        return result;
    }

    std::string header_line;
    if (!std::getline(file, header_line))
    {
        std::cerr << "CSV file is empty or cannot read header: " << csv_file << std::endl;
        result.row_count = 0;
        result.has_more = false;
        return result;
    }

    // Parse header to get all column names
    std::vector<std::string> all_columns;
    std::stringstream header_ss(header_line);
    std::string col_name;
    while (std::getline(header_ss, col_name, ','))
    {
        // Trim whitespace from col_name
        col_name.erase(0, col_name.find_first_not_of(" \t\r\n"));
        col_name.erase(col_name.find_last_not_of(" \t\r\n") + 1);

        // Remove suffixes like " (N)", " (P)", " (D)" from the column name
        size_t paren_pos = col_name.find(" (");
        if (paren_pos != std::string::npos)
        {
            col_name = col_name.substr(0, paren_pos);
            // Trim any trailing whitespace after removing suffix
            col_name.erase(col_name.find_last_not_of(" \t\r\n") + 1);
        }

        all_columns.push_back(col_name);
    }

    // Determine which columns to read
    std::vector<std::string> cols_to_read = this->projections;
    if (cols_to_read.empty())
    {
        cols_to_read = all_columns;
    }

    // Map projection columns to column indexes in CSV file
    std::vector<int> col_indexes;
    for (const auto &proj_col : cols_to_read)
    {
        auto it = std::find(all_columns.begin(), all_columns.end(), proj_col);
        if (it == all_columns.end())
        {
            std::cerr << "Column " << proj_col << " not found in CSV header" << std::endl;
            // You might want to handle this differently, skipping or error out
            continue;
        }
        col_indexes.push_back(std::distance(all_columns.begin(), it));
    }

    // Column metadata setup (reuse your existing logic)
    int idx = 0;
    for (const auto &col_name : cols_to_read)
    {
        ColumnInfo *c = data_base->get_column(table_name, col_name);
        c->idx = idx++;
        if (!c)
        {
            std::cerr << "Column " << col_name << " not found in table metadata " << table_name << std::endl;
            continue;
        }
        result.columns.push_back(*c);
    }
    result.column_count = result.columns.size();

    // Skip rows before the batch start
    size_t start_row = batch_index * batch_size;
    size_t skipped = 0;
    std::string dummy_line;
    while (skipped < start_row && std::getline(file, dummy_line))
    {
        ++skipped;
    }

    // Read rows for batch
    size_t rows_read = 0;
    std::vector<std::vector<std::string>> batch_rows; // temporarily hold strings
    batch_rows.reserve(batch_size);
    while (rows_read < batch_size && std::getline(file, dummy_line))
    {
        // Parse CSV line into columns
        std::stringstream line_ss(dummy_line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(line_ss, cell, ','))
        {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            cells.push_back(cell);
        }

        // Select projection columns only
        std::vector<std::string> proj_cells;
        for (int ci : col_indexes)
        {
            if (ci < (int)cells.size())
                proj_cells.push_back(cells[ci]);
            else
                proj_cells.push_back("");
        }

        batch_rows.push_back(proj_cells);
        ++rows_read;
    }
    result.row_count = batch_rows.size();

    // Determine if there are more rows left
    bool has_more = false;
    if (rows_read == batch_size && std::getline(file, dummy_line))
    {
        has_more = true;
    }
    result.has_more = has_more;

    if (result.row_count == 0)
    {
        return result;
    }

    // Allocate memory for each column's data for the batch
    result.data.resize(result.column_count);

    for (size_t col_idx = 0; col_idx < result.columns.size(); ++col_idx)
    {
        ColumnInfo &col = result.columns[col_idx];
        switch (col.type)
        {
        case DataType::FLOAT:
        {
            float *data_ptr = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                const std::string &val_str = batch_rows[row_idx][col_idx];
                try
                {
                    if (val_str.empty())
                        throw std::runtime_error("empty string");
                    data_ptr[row_idx] = std::stof(val_str);
                }
                catch (...)
                {
                    data_ptr[row_idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
            result.data[col_idx] = data_ptr;
            break;
        }
        case DataType::DATETIME:
        {
            uint64_t *data_ptr = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                const std::string &val_str = batch_rows[row_idx][col_idx];
                try
                {
                    if (val_str.empty())
                        throw std::runtime_error("empty string");
                    data_ptr[row_idx] = getDateTime(val_str);
                }
                catch (...)
                {
                    const double nan_value = std::numeric_limits<double>::quiet_NaN();
                    data_ptr[row_idx] = *reinterpret_cast<const uint64_t *>(&nan_value);
                }
            }
            result.data[col_idx] = data_ptr;
            break;
        }
        case DataType::STRING:
        {
            char **data_ptr = static_cast<char **>(malloc(result.row_count * sizeof(char *)));
            for (size_t row_idx = 0; row_idx < result.row_count; ++row_idx)
            {
                const std::string &val_str = batch_rows[row_idx][col_idx];
                if (!val_str.empty())
                {
                    data_ptr[row_idx] = strdup(val_str.c_str());
                }
                else
                {
                    data_ptr[row_idx] = nullptr;
                }
            }
            result.data[col_idx] = data_ptr;
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type in column processing");
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
    else if (c->type == DataType::FLOAT)
    {
        new_col_name = col_name + " (N)";
    }

    if (c->is_primary)
    {
        new_col_name += " (P)";
    }
    return new_col_name;
}
