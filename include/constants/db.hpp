#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <variant>
#include <fstream>
#include "data_types.hpp"
#include <iomanip> // For std::fixed, std::setprecision

struct ColumnInfo
{
    std::string name;
    DataType type;
    uint8_t idx;
    bool is_primary;
    ColumnInfo() : is_primary(false) {}
};

struct Table
{
    std::vector<ColumnInfo> cols;
    std::string name;
    uint64_t size;

    std::vector<std::string> getColumnNames()
    {
        std::vector<std::string> col_names;
        for (const auto &col : cols)
        {
            col_names.push_back(col.name);
        }
        return col_names;
    }
};

class TableResults
{
public:
    TableResults() {}
    ~TableResults() {}
    std::vector<ColumnInfo> columns;
    std::vector<void *> data;
    size_t row_count = 0;
    size_t column_count = 0;
    bool has_more = true;
    bool is_join = false;
    bool end_right = false;
    size_t batch_index = 0;
    size_t batch_index_right = 0;
    size_t total_rows = 0;

    size_t estimateMemorySize() const
    {
        size_t total_size = 0;
        for (size_t i = 0; i < column_count; ++i)
        {
            total_size += row_count * getDataTypeNumBytes(columns[i].type);
        }
        return total_size;
    }

    size_t
    getColumnIndex(const std::string &column_name) const
    {
        for (size_t i = 0; i < columns.size(); ++i)
        {
            if (columns[i].name == column_name)
            {
                return i;
            }
        }
        throw std::runtime_error("Column not found: " + column_name);
    }
    void print() const
    {
        for (const auto &col : columns)
        {
            std::cout << col.name << "\t";
        }
        std::cout << "\n";

        // Print a separator line
        for (size_t i = 0; i < columns.size(); ++i)
        {
            std::cout << "--------";
        }
        std::cout << "\n";

        for (size_t row = 0; row < row_count; ++row)
        {
            for (size_t col = 0; col < column_count; ++col)
            {

                if (columns[col].type == DataType::FLOAT)
                {
                    std::cout << static_cast<float *>(data[col])[row] << "\t";
                }
                else if (columns[col].type == DataType::DATETIME)
                {
                    std::cout << static_cast<uint64_t *>(data[col])[row] << "\t";
                }
                else if (columns[col].type == DataType::STRING)
                {
                    std::cout << static_cast<char **>(data[col])[row] << "\t";
                }
            }
            std::cout << "\n";
        }
        std::cout << "Total Rows: " << row_count << "\n";
        std::cout << "Total Columns: " << column_count << "\n";
        std::cout << "---------------------------------\n";
    }
    bool is_file_empty(const std::string &filename) const
    {
        std::ifstream in_file(filename, std::ios::binary | std::ios::ate);
        if (!in_file.is_open())
        {
            return true;
        }
        auto size = in_file.tellg();
        in_file.close();
        return size == 0;
    }
    void write_to_file(std::string query_file_name) const
    {
        std::string query_name = query_file_name;
        size_t start_pos = query_name.find("query");
        size_t end_pos = query_name.find(".txt");
        if (start_pos != std::string::npos && end_pos != std::string::npos && start_pos < end_pos)
        {
            query_name = query_name.substr(start_pos, end_pos - start_pos);
        }
        std::string filename = "Team9_" + query_name + ".csv";
        std::ofstream file(filename, std::ios::app); // Open file in append mode
        if (!file.is_open())
        {
            // If the file doesn't exist, create it
            std::ofstream create_file(filename);
            if (!create_file.is_open())
            {
            std::cerr << "Error creating file: " << filename << "\n";
            return;
            }
            create_file.close();
            file.open(filename, std::ios::app); // Reopen in append mode
        }
        // Write column headers only if it's the first batch
        if (is_file_empty(filename))
        {
            // Write column headers
            for (size_t col = 0; col < column_count; ++col)
            {
                file << columns[col].name;
                if (col < column_count - 1)
                    file << ",";
            }
            file << "\n";
        }

        // Write data rows
        for (size_t row = 0; row < row_count; ++row)
        {
            for (size_t col = 0; col < column_count; ++col)
            {
                if (columns[col].type == DataType::FLOAT)
                {
                    // Convert float to string with fixed precision
                    file << std::fixed << std::setprecision(2) << static_cast<float *>(data[col])[row];
                }
                else if (columns[col].type == DataType::DATETIME)
                {
                    uint64_t timestamp = static_cast<uint64_t *>(data[col])[row];
                    // std::time_t time = static_cast<std::time_t>(timestamp);
                    file << getDateTimeStr(timestamp);
                }
                else if (columns[col].type == DataType::STRING)
                {
                    std::string str = "";
                    if (static_cast<char **>(data[col])[row])
                    {
                        str = static_cast<char **>(data[col])[row];
                    }

                    if (str.find(',') != std::string::npos)
                    {
                        file << "\"" << str << "\"";
                    }
                    else
                    {
                        file << str;
                    }
                }
                if (col < column_count - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
        // std::cout << "Data appended to file: " << filename << "\n";
    }
};

class DB
{
public:
    DB() {}
    ~DB() {}
    std::vector<Table> tables;

    void print_databse()
    {

        for (const auto &table : tables)
        {
            std::cout << "Table Name: " << table.name << "\n";
            std::cout << "Columns:\n";
            for (const auto &col : table.cols)
            {
                std::cout << "  - " << col.name << " (" << getDataTypeString(col.type) << ")\n";
            }
            std::cout << "\n";
        }
    }
    Table *get_table(const std::string &table_name)
    {
        for (auto &table : tables)
        {
            if (table.name == table_name)
            {
                return &table;
            }
        }
        return nullptr;
    }

    ColumnInfo *get_column(const std::string &table_name, const std::string &col_name)
    {
        Table *table = get_table(table_name);
        if (table)
        {
            for (auto &col : table->cols)
            {
                if (col.name == col_name)
                {
                    return &col;
                }
            }
        }
        return nullptr;
    }
};