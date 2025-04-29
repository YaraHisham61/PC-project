#pragma once
#include <vector>
#include "duckdb.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <vector>
#include <filesystem>
#include <string>
#include "constants/db.hpp"
#include "csv_parser/rapidcsv.h"


namespace fs = std::filesystem;

class CSVImporter
{
public:
    CSVImporter(const duckdb::DuckDB &db, duckdb::Connection &con);
    ~CSVImporter();
    bool import_folder(const std::string &folder_path, DB *data_base);
    TableResults read_table(DB *data_base, const std::string &table_name, const std::vector<std::string> &cols = {});
    const duckdb::DuckDB &db;
    duckdb::Connection &con;

private:
    std::vector<ColumnInfo> parse_csv_header(const std::string &line) const;
    bool import_csv(const std::string &csv_file, DB *data_base, const std::string &table_name = "my_table");
    void process_column_token(std::string token, std::vector<ColumnInfo> &columns, size_t idx) const;
    void trim(std::string &s) const;
    DataType map_column_type(const std::string &type_info) const;
    std::string get_table_name(const fs::path &file_path) const;
    std::string get_original_column_name(const ColumnInfo *c, const std::string &col_name) const;
};

// #endif // CSV_IMPORTER_H