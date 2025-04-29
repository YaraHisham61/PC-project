#include "dbms/duckdb_interface.hpp"

std::unique_ptr<duckdb::LogicalOperator> DuckDBInterface::getLogicalPlan(const std::string &query)
{
    duckdb::Parser parser;
    parser.ParseQuery(query);
    auto statements = std::move(parser.statements);
    duckdb::Planner planner(*con.context);
    planner.CreatePlan(std::move(statements[0]));
    duckdb::Optimizer optimizer(*planner.binder, *con.context);
    auto logical_plan = std::move(optimizer.Optimize(std::move(planner.plan)));
    return std::move(logical_plan);
}

// std::unique_ptr<duckdb::PhysicalOperator> DuckDBInterface::getPhysicalPlan(auto logical_plan)
// {
//     duckdb::PhysicalPlanGenerator physical_plan_generator(*con.context);
//     auto physical_plan = physical_plan_generator.Plan(logical_plan);
//     return physical_plan;
//     // duckdb::PhysicalPlanGenerator physical_plan_generator(*con.context);
//     // auto physical_plan = physical_plan_generator.Plan(logical_plan->Copy(*con.context));
//     // std::cout << "Physical plan:\n";
// }

void DuckDBInterface::printPhysicalPlan(duckdb::PhysicalOperator *op, int indent)
{
    if (!op)
        return;

    std::string pad(indent, ' ');
    std::cout << pad << "- " << op->GetName() << "\n";
    auto m = op->ParamsToString();

    // if (op->type == duckdb::PhysicalOperatorType::TABLE_SCAN){
    //     auto &table_scan = op->Cast<duckdb::PhysicalTableScan>();
    // }
    // else if (op->type == duckdb::PhysicalOperatorType::PROJECTION)
    // {
    //     auto &projection = op->Cast<duckdb::PhysicalProjection>();
    //     std::cout << pad << "  Projection: ";
    //     for (auto &expr : projection.expressions)
    //     {
    //         std::cout << expr->ToString() << ", ";
    //     }
    //     std::cout << "\n";
    // }

    if (m.size() > 0)
    {
        std::cout << pad << "  Params: ";
        for (auto &pair : m)
        {
            std::cout << pair.first << ": " << pair.second << ", ";
        }
        std::cout << "\n";
    }
    for (auto &child : op->children)
    {
        printPhysicalPlan(&(child.get()), indent + 2);
    }
}