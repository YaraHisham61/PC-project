#pragma once

#include "physical_plan/physical_op.hpp"

#include "kernels/hash_join.hpp"
#include <cuda_runtime.h>
#include <sstream>
class HashJoin : public PhysicalOpNode
{
    std::string col_table_left;
    std::string col_table_right;

public:
    HashJoin(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    ~HashJoin() override = default;
    TableResults executeJoin(const TableResults &left_table, const TableResults &right_table);
    void print() const override;
};
