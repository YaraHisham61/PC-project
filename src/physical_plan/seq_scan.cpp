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
    // Load CSV file
    std::string csv_file = DATA_DIR + this->table_name + ".csv";
    this->doc = rapidcsv::Document(csv_file);
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

    std::vector<std::string> cols_to_read = this->projections;

    size_t total_rows = this->doc.GetRowCount();
    size_t start_row = batch_index * batch_size;
    if (start_row >= total_rows)
    {
        result.row_count = 0;
        result.has_more = false; // No more batches to read
        return result;
    }
    size_t end_row = std::min(start_row + batch_size, total_rows);
    result.row_count = end_row - start_row;
    result.has_more = (end_row < total_rows);

    // If no projections specified, read all columns
    if (cols_to_read.empty())
    {
        cols_to_read = data_base->get_table(table_name)->getColumnNames();
    }

    // Set up column metadata
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

    if (result.row_count == 0)
    {
        result.has_more = false;
        return result; // No rows to read
    }

    // Allocate memory for the batch
    result.data.resize(result.column_count);

    // Read data for each column
    for (const auto &col : result.columns)
    {
        std::string col_name = get_original_column_name(&col, cols_to_read[col.idx]);

        switch (col.type)
        {
        case DataType::FLOAT:
            result.data[col.idx] = static_cast<float *>(malloc(result.row_count * sizeof(float)));
            for (size_t row_idx = start_row; row_idx < end_row; ++row_idx)
            {
                try
                {
                    float value = this->doc.GetCell<float>(col_name, row_idx);
                    static_cast<float *>(result.data[col.idx])[row_idx - start_row] = value;
                }
                catch (const std::exception &e)
                {
                    // Use NaN for null float values
                    static_cast<float *>(result.data[col.idx])[row_idx - start_row] = std::numeric_limits<float>::quiet_NaN();
                }
            }
            break;

        case DataType::DATETIME:
            result.data[col.idx] = static_cast<uint64_t *>(malloc(result.row_count * sizeof(uint64_t)));
            for (size_t row_idx = start_row; row_idx < end_row; ++row_idx)
            {
                try
                {
                    std::string date_str = this->doc.GetCell<std::string>(col_name, row_idx);
                    static_cast<uint64_t *>(result.data[col.idx])[row_idx - start_row] = getDateTime(date_str);
                }
                catch (const std::exception &e)
                {
                    static_cast<uint64_t *>(result.data[col.idx])[row_idx - start_row] = 0;
                }
            }
            break;

        case DataType::STRING:
            result.data[col.idx] = static_cast<char **>(malloc(result.row_count * sizeof(char *)));
            for (size_t row_idx = start_row; row_idx < end_row; ++row_idx)
            {
                try
                {
                    std::string str_value = this->doc.GetCell<std::string>(col_name, row_idx);
                    static_cast<char **>(result.data[col.idx])[row_idx - start_row] = strdup(str_value.c_str());
                }
                catch (const std::exception &e)
                {
                    static_cast<char **>(result.data[col.idx])[row_idx - start_row] = nullptr;
                }
            }
            break;

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
