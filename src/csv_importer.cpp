#include "dbms/csv_importer.hpp"

CSVImporter::CSVImporter(const duckdb::DuckDB &db, duckdb::Connection &con)
    : db(db), con(con)
{

    // con.Query("SET disabled_optimizers = 'filter_pushdown, statistics_propagation';");
}

CSVImporter::~CSVImporter()
{
}

bool CSVImporter::import_folder(const std::string &folder_path, DB *data_base)
{
    try
    {
        size_t imported_count = 0;

        for (const auto &entry : fs::directory_iterator(folder_path))
        {
            if (entry.path().extension() == ".csv")
            {
                std::string table_name = get_table_name(entry.path());
                if (import_csv(entry.path(), data_base, table_name))
                {
                    imported_count++;
                    std::cout << "Imported " << entry.path().filename()
                              << " as table '" << table_name << "'\n";
                }
            }
        }

        std::cout << "Successfully imported " << imported_count << " tables\n";
        return imported_count > 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error importing folder: " << e.what() << std::endl;
        return false;
    }
}

bool CSVImporter::import_csv(const std::string &csv_file, DB *data_base, const std::string &table_name)
{
    try
    {
        Table t = Table();
        t.name = table_name;
        // 1. Get column info from CSV header
        std::ifstream file(csv_file);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + csv_file);
        }

        std::string header_line;
        if (!getline(file, header_line))
        {
            throw std::runtime_error("Empty file or could not read header");
        }

        std::vector<ColumnInfo> columns = parse_csv_header(header_line);
        t.cols = columns;
        // 2. Create table with proper schema
        std::string create_sql = "CREATE TABLE " + table_name + " (";
        for (size_t i = 0; i < columns.size(); i++)
        {
            if (i != 0)
                create_sql += ", ";
            create_sql += "\"" + columns[i].name + "\" " + getDataTypeString(columns[i].type);
            if (columns[i].is_primary)
            {
                create_sql += " PRIMARY KEY";
            }
        }
        create_sql += ")";

        auto create_result = con.Query(create_sql);
        if (create_result->HasError())
        {
            throw std::runtime_error("Table creation failed: " + create_result->GetError());
        }
        data_base->tables.push_back(t);
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}
std::string CSVImporter::get_table_name(const fs::path &file_path) const
{
    return file_path.stem().string(); // Returns filename without extension
}

std::string CSVImporter::get_original_column_name(const ColumnInfo *c, const std::string &col_name) const
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

std::vector<ColumnInfo> CSVImporter::parse_csv_header(const std::string &line) const
{
    std::vector<ColumnInfo> columns;
    size_t start = 0;
    size_t end = line.find(',');
    size_t idx = 0;

    while (end != std::string::npos)
    {

        std::string token = line.substr(start, end - start);
        process_column_token(token, columns, idx++);
        start = end + 1;
        end = line.find(',', start);
    }

    process_column_token(line.substr(start), columns, idx);

    return columns;
}

void CSVImporter::process_column_token(std::string token, std::vector<ColumnInfo> &columns, size_t idx) const
{
    ColumnInfo col;
    size_t paren_start = token.find('(');

    // Extract column name
    col.name = token.substr(0, paren_start);
    trim(col.name);

    // Extract type info inside parentheses
    size_t paren_end = token.find(')', paren_start);
    if (paren_start != std::string::npos && paren_end != std::string::npos)
    {
        std::string type_info = token.substr(paren_start + 1, paren_end - paren_start - 1);
        col.type = map_column_type(type_info);
        col.is_primary = token.find('P') != std::string::npos;
        col.idx = idx;
    }
    else
    {
        // Default to VARCHAR if no type specified
        col.type = DataType::STRING;
    }

    columns.push_back(col);
}

DataType CSVImporter::map_column_type(const std::string &type_info) const
{
    if (type_info.find('N') != std::string::npos)
    {
        return DataType::FLOAT;
    }
    else if (type_info.find('D') != std::string::npos)
    {
        return DataType::DATETIME;
    }
    return DataType::STRING; // Default type
}

void CSVImporter::trim(std::string &s) const
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch)
                                    { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch)
                         { return !std::isspace(ch); })
                .base(),
            s.end());
}