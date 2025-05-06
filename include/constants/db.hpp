#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <variant>
#include <fstream>
#include "data_types.hpp"
#include <iomanip> // For std::fixed, std::setprecision

// Struct for column metadata

#define DATA_DIR "/mnt/e/Collage/PC - Parallel Computing/Project/dbms/data/"

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
    size_t batch_index = 0;

    size_t getColumnIndex(const std::string &column_name) const
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
    void print()
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
    bool is_file_empty(const std::string &filename)
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
    void write_to_file()
    {
        std::string filename = std::string(DATA_DIR) + "output.csv";
        std::ofstream file(filename, std::ios::app); // Open file in append mode
        if (!file.is_open())
        {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }

        // Write column headers only if it's the first batch
        if (this->batch_index == 0 || is_file_empty(filename))
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
                    std::string str = static_cast<char **>(data[col])[row];
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
    void write_aggregate_to_file()
    {
        std::string filename = std::string(DATA_DIR) + "output.txt";
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file for writing: " << filename << "\n";
            return;
        }
        // Write column headers
        for (size_t col = 0; col < column_count; ++col)
        {
            file << columns[col].name;
            if (col < column_count - 1)
                file << "\t";
        }
        file << "\n";
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
                    std::time_t time = static_cast<std::time_t>(timestamp);
                    file << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
                }
                else if (columns[col].type == DataType::STRING)
                {
                    std::string str = static_cast<char **>(data[col])[row];
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
                    file << "\t";
            }
            file << "\n";
        }
        file.close();
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