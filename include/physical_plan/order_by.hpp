#pragma once

#include "physical_plan/physical_op.hpp"

#include "kernels/order_by.hpp"
#include <cuda_runtime.h>
#include <sstream>

class OrderBy : public PhysicalOpNode
{
    std::string table_name;
    std::string col_name;
    bool is_Ascending;

public:
    OrderBy(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    ~OrderBy() override = default;
    TableResults executeOrderBy(const TableResults &input_table);
    void print() const override;
};
