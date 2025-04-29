#pragma once
#include "duckdb.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include <iostream>

class DuckDBInterface
{
private:
    const duckdb::DuckDB &db;
    duckdb::Connection &con;

public:
    DuckDBInterface(const duckdb::DuckDB &db, duckdb::Connection &con) : db(db), con(con) {}
    ~DuckDBInterface() {}
    std::unique_ptr<duckdb::LogicalOperator> getLogicalPlan(const std::string &query);
    // std::unique_ptr<duckdb::PhysicalOperator> getPhysicalPlan(auto logical_plan);
    void printPhysicalPlan(duckdb::PhysicalOperator *op, int indent = 0);
};