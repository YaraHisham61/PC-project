#pragma once

#include "physical_plan/physical_op.hpp"
#include "kernels/filter_kernel.hpp"
#include <cuda_runtime.h>
#include <regex>
#include <cctype>
class NestedLoopJoin : public PhysicalOpNode
{
    std::string op;
    std::string column;

public:
    NestedLoopJoin(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    TableResults applyNested(const TableResults &left_table, const TableResults &input_table) const;
    bool *getSelectedRows(const TableResults &input_table, float value) const;
    std::string extract_base_column_name(std::string column_name);
    void print() const override;
};