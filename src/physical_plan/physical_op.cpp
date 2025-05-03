#include "physical_plan/physical_op.hpp"
#include "physical_plan/seq_scan.hpp"
#include "physical_plan/projection.hpp"
#include "physical_plan/filter.hpp"
#include "physical_plan/join.hpp"
#include "physical_plan/aggregate.hpp"
std::unique_ptr<PhysicalOpNode> PhysicalOpNode::buildPlanTree(
    duckdb::PhysicalOperator *op,
    DB *data_base,
    TableResults **input_table_ptr)
{
    if (!op)
    {
        return nullptr;
    }

    std::unique_ptr<PhysicalOpNode> node;
    std::string op_name = op->GetName();
    auto params = op->ParamsToString();
    op_name.erase(std::remove(op_name.begin(), op_name.end(), ' '), op_name.end());

    // First create the current node
    if (op_name == "SEQ_SCAN")
    {
        node = std::make_unique<SeqScan>(params);
    }
    else if (op_name == "FILTER")
    {
        node = std::make_unique<Filter>(params);
    }
    else if (op_name == "PROJECTION")
    {
        node = std::make_unique<Projection>(params);
    }
    else if (op_name == "UNGROUPED_AGGREGATE")
    {
        node = std::make_unique<Aggregate>(params);
    }

    else if (op_name == "HASH_JOIN")
    {
        node = std::make_unique<HashJoin>(params);
    }

    if (op_name == "HASH_JOIN")
    {
        auto *join_ptr = static_cast<HashJoin *>(node.get());

        TableResults *left_table_ptr = nullptr;
        auto left_child = buildPlanTree(&(op->children[0].get()), data_base, &left_table_ptr);

        // Process right child
        TableResults *right_table_ptr = nullptr;
        auto right_child = buildPlanTree(&(op->children[1].get()), data_base, &right_table_ptr);

        if (left_child)
            node->children.push_back(std::move(left_child));
        if (right_child)
            node->children.push_back(std::move(right_child));

        if (!left_table_ptr || !right_table_ptr)
        {
            std::cerr << "Error: Missing input tables for join\n";
            return nullptr;
        }

        TableResults join_result = join_ptr->executeJoin(*left_table_ptr, *right_table_ptr);
        join_result.print();
        // delete left_table_ptr;
        // delete right_table_ptr;

        if (*input_table_ptr)
        {
            **input_table_ptr = std::move(join_result);
        }
        else
        {
            *input_table_ptr = new TableResults(std::move(join_result));
        }

        return node;
    }
    for (auto &child : op->children)
    {
        auto child_node = buildPlanTree(&(child.get()), data_base, input_table_ptr);
        if (child_node && !child_node->params.empty())
        {
            node->children.push_back(std::move(child_node));
        }
    }

    // Now execute the current operation
    if (op_name == "SEQ_SCAN")
    {
        auto *seq_ptr = static_cast<SeqScan *>(node.get());
        TableResults scan_result = seq_ptr->read_scan_table(data_base);
        // scan_result.print();
        if (*input_table_ptr)
        {
            **input_table_ptr = std::move(scan_result);
        }
        else
        {
            *input_table_ptr = new TableResults(std::move(scan_result));
        }
        // input_table_ptr->print();
    }
    else if (op_name == "FILTER")
    {
        if (!*input_table_ptr)
        {
            std::cerr << "Error: No input table to filter\n";
            return nullptr;
        }

        auto *filter_ptr = static_cast<Filter *>(node.get());
        TableResults filtered_result = filter_ptr->applyFilter(**input_table_ptr);
        // filtered_result.print();
        **input_table_ptr = std::move(filtered_result);
    }
    else if (op_name == "PROJECTION")
    {
        if (!*input_table_ptr)
        {
            std::cerr << "Error: No input table to project\n";
            return nullptr;
        }

        auto *proj_ptr = static_cast<Projection *>(node.get());
        TableResults projected_result = proj_ptr->applyProjection(**input_table_ptr);
        projected_result.print();
        **input_table_ptr = std::move(projected_result);
    }
    else if (op_name == "UNGROUPED_AGGREGATE")
    {
        if (!*input_table_ptr)
        {
            std::cerr << "Error: No input table to project\n";
            return nullptr;
        }

        auto *aggr_ptr = static_cast<Aggregate *>(node.get());
        TableResults aggregate_result = aggr_ptr->computeAggregates(**input_table_ptr);
        aggregate_result.print();
        **input_table_ptr = std::move(aggregate_result);
    }

    return node;
}