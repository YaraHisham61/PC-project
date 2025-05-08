#pragma once

#include "physical_plan/physical_op.hpp"

#include "kernels/order_by_kernel.hpp"
#include "kernels/hash_join_kernel.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <filesystem>
#include <fstream>

class OrderBy : public PhysicalOpNode
{
    std::string table_name;
    std::string col_name;
    bool is_Ascending;

public:
    OrderBy(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    ~OrderBy() override = default;
    std::vector<size_t> getSortedIndex(const TableResults &input_table);
    TableResults executeOrderBy(const TableResults &input_table);
    TableResults mergeSortedBatchesOnGPU(const std::vector<TableResults> &batches);
    void write_intermideate(const std::vector<TableResults> &batches);
    TableResults read_intermediate(const std::string &filename);
    TableResults merge_sorted_files();
    void print() const override;
};
