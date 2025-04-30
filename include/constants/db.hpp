#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <variant>
#include "data_types.hpp"

// Struct for column metadata

#define DATA_DIR "/mnt/e/Collage/PC - Parallel Computing/Project/dbms/data/"
// #define DATA_DIR "/mnt/c/Users/LENOVE/Desktop/second term 4/p/PC-project/data/"

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

    size_t getColumnIndex(const std::string &column_name) const
    {
        for (size_t i = 0; i < columns.size(); ++i)
        {
            if (columns[i].name == column_name)
            {
                return i;
            }
        }
        return -1;
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
                if (columns[col].type == DataType::INT)
                {
                    std::cout << static_cast<int *>(data[col])[row] << "\t";
                }
                else if (columns[col].type == DataType::FLOAT)
                {
                    std::cout << static_cast<float *>(data[col])[row] << "\t";
                }
                else if (columns[col].type == DataType::DATETIME)
                {
                    std::cout << static_cast<int64_t *>(data[col])[row] << "\t";
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