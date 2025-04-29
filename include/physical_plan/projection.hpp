#pragma once

#include "physical_plan/physical_op.hpp"

class Projection : public PhysicalOpNode
{
public:
    bool flag = false;
    std::vector<int> projections_index;
    std::vector<std::string> output_names;
    Projection(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    TableResults applyProjection(const TableResults &input_table) const;
    void parseProjectionList(const std::string &projection_list);
    void print() const override;
    ~Projection() = default;
};