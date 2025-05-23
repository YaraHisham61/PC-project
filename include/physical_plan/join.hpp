#pragma once

#include "physical_plan/physical_op.hpp"

#include "kernels/hash_join_kernel.hpp"
#include <cuda_runtime.h>
#include <sstream>
class HashJoin : public PhysicalOpNode
{
    std::string col_table_left;
    std::string col_table_right;

public:
    HashJoin(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    ~HashJoin() override = default;
    void getIndexOfSelectedRows(const TableResults &left_table, const TableResults &right_table,
                                std::vector<size_t> &left_indices, std::vector<size_t> &right_indices);
    void getIndexOfSelectedRowsCPU(const TableResults &left_table, const TableResults &right_table,
                                   std::vector<size_t> &left_indices, std::vector<size_t> &right_indices);
    TableResults executeJoin(const TableResults &left_table, const TableResults &right_table);
    TableResults executeJoinCPU(const TableResults &left_table, const TableResults &right_table);
    void print() const override;
};
