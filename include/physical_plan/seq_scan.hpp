#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "physical_plan/physical_op.hpp"

class SeqScan : public PhysicalOpNode
{
public:
    std::string table_name;
    std::vector<std::string> projections;

    SeqScan(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    ~SeqScan() = default;
    TableResults read_scan_table(DB *data_base, size_t batch_index, size_t batch_size);
    std::string get_original_column_name(const ColumnInfo *c, const std::string &col_name);
    void print() const override;
};