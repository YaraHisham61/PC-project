#include "physical_plan/physical_op.hpp"
#include "physical_plan/seq_scan.hpp"
#include "physical_plan/projection.hpp"
#include "physical_plan/filter.hpp"
#include "physical_plan/join.hpp"
#include "physical_plan/aggregate.hpp"
#include "physical_plan/order_by.hpp"
#include "dbms/profiler.hpp"

std::unique_ptr<PhysicalOpNode> PhysicalOpNode::buildPlanTree(
    duckdb::PhysicalOperator *op,
    DB *data_base,
    TableResults **input_table_ptr,
    size_t batch_index, size_t batch_size, size_t batch_index_right, bool *is_join, bool *end_right, bool GPU)
{
    if (!op)
    {
        return nullptr;
    }

    Profiler profiler;
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
        // (*input_table_ptr)->is_join = true;

        node = std::make_unique<HashJoin>(params);
    }
    else if (op_name == "ORDER_BY")
    {
        node = std::make_unique<OrderBy>(params);
    }

    if (op_name == "HASH_JOIN")
    {
        auto *join_ptr = static_cast<HashJoin *>(node.get());
        *is_join = true;
        TableResults *left_table_ptr = nullptr;
        auto left_child = buildPlanTree(&(op->children[0].get()), data_base, &left_table_ptr, batch_index, batch_size, batch_index_right, is_join, end_right, GPU);
        if (left_table_ptr->row_count == 0)
        {
            return node;
        }

        if (left_child)
            node->children.push_back(std::move(left_child));

        TableResults *right_table_ptr = nullptr;
        auto right_child = buildPlanTree(&(op->children[1].get()), data_base, &right_table_ptr, batch_index_right, batch_size, batch_index_right, is_join, end_right, GPU);

        if (right_child)
            node->children.push_back(std::move(right_child));
        if (!left_table_ptr || !right_table_ptr)
        {
            std::cerr << "Error: Missing input tables for join\n";
            return nullptr;
        }
        TableResults join_result;
        if (GPU)
        {
            profiler.start("GPU Join");
            join_result = join_ptr->executeJoin(*left_table_ptr, *right_table_ptr);
            profiler.stop("GPU Join");
        }
        else
        {
            profiler.start("CPU Join");
            join_result = join_ptr->executeJoinCPU(*left_table_ptr, *right_table_ptr);
            profiler.stop("CPU Join");
        }
        if (*input_table_ptr)
        {
            **input_table_ptr = std::move(join_result);
        }
        else
        {
            *input_table_ptr = new TableResults(std::move(join_result));
        }
        (*input_table_ptr)->is_join = true;
        if (right_table_ptr->has_more == false)
        {
            *end_right = true;
        }
        else
        {
            *end_right = false;
        }
        return node;
    }
    for (auto &child : op->children)
    {
        auto child_node = buildPlanTree(&(child.get()), data_base, input_table_ptr, batch_index, batch_size, batch_index_right, is_join, end_right, GPU);
        if (child_node && !child_node->params.empty())
        {
            node->children.push_back(std::move(child_node));
        }
    }

    // Now execute the current operation
    if (op_name == "SEQ_SCAN")
    {
        auto *seq_ptr = static_cast<SeqScan *>(node.get());
        TableResults scan_result = seq_ptr->read_scan_table(data_base, batch_index, batch_size);
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
        if (GPU)
        {
            profiler.start("GPU Filter");
            TableResults filtered_result = filter_ptr->applyFilter(**input_table_ptr);
            **input_table_ptr = std::move(filtered_result);
            profiler.stop("GPU Filter");
        }
        else
        {
            profiler.start("CPU Filter");
            TableResults filtered_result = filter_ptr->applyFilterCPU(**input_table_ptr);
            **input_table_ptr = std::move(filtered_result);
            profiler.stop("CPU Filter");
        }
    }
    else if (op_name == "PROJECTION")
    {
        if (!*input_table_ptr)
        {
            std::cerr << "Error: No input table to project\n";
            return nullptr;
        }

        auto *proj_ptr = static_cast<Projection *>(node.get());
        if (proj_ptr->flag == true)
        {
            if ((op->children[0].get().GetName()) == "HASH_JOIN")
                return node;
        }
        TableResults projected_result = proj_ptr->applyProjection(**input_table_ptr);
        // projected_result.print();
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
        // aggregate_result.print();
        **input_table_ptr = std::move(aggregate_result);
    }

    else if (op_name == "ORDER_BY")
    {
        if (!*input_table_ptr)
        {
            std::cerr << "Error: No input table to project\n";
            return nullptr;
        }

        auto *order_ptr = static_cast<OrderBy *>(node.get());
        TableResults ordered_result = order_ptr->executeOrderBy(**input_table_ptr);
        // ordered_result.print();
        **input_table_ptr = std::move(ordered_result);
    }

    return node;
}

void PhysicalOpNode::executePlanInBatches(
    duckdb::PhysicalOperator *op, DB *data_base, size_t batch_size = 1000, bool GPU = true)
{
    TableResults *current_batch = nullptr;
    size_t batch_index = 0;
    size_t batch_index_right = 0;
    size_t total_rows = 0;
    bool has_more = true;

    bool is_aggregate = (op->GetName() == "UNGROUPED_AGGREGATE");
    bool is_order_by = (op->GetName() == "ORDER_BY");
    std::vector<TableResults> order_by_batches; // For OrderBy merging

    std::unique_ptr<Aggregate> aggregate_op;
    std::unique_ptr<OrderBy> order_by_op;
    if (is_aggregate)
    {
        aggregate_op = std::make_unique<Aggregate>(op->ParamsToString());
    }
    else if (is_order_by)
    {
        order_by_op = std::make_unique<OrderBy>(op->ParamsToString());
    }
    Profiler profiler;
    bool is_join = false;
    bool end_right = false;
    while (true)
    {
        current_batch = nullptr;
        profiler.start("JOIN");
        auto plan_tree = buildPlanTree(op, data_base, &current_batch, batch_index, batch_size, batch_index_right, &is_join, &end_right, GPU);
        profiler.stop("JOIN");
        current_batch->is_join = is_join;
        current_batch->end_right = end_right;
        if (!current_batch)
        {
            std::cerr << "Error: No result for batch " << batch_index << "\n";
            break;
        }
        if (is_join)
        {
            has_more = current_batch->has_more || !end_right;
        }
        else
        {
            has_more = current_batch->has_more;
        }

        if (is_aggregate && current_batch->row_count != 0)
        {
            aggregate_op->updateAggregates(*current_batch);
            total_rows += current_batch->total_rows;
        }
        else if (is_order_by)
        {
            order_by_batches.push_back(*current_batch);
        }
        else
        {

            // current_batch->batch_index = batch_index;
            if (current_batch->row_count != 0)
            {
                current_batch->write_to_file();
                std::cout << "Batch " << batch_index << "   " << current_batch->row_count << ":\n";
            }
        }

        if (!has_more)
            break;
        if (is_join)
        {
            batch_index_right++;
            current_batch->batch_index_right = batch_index_right;
            if (end_right)
            {
                current_batch->end_right = false;
                end_right = false;
                batch_index_right = 0;
                batch_index++;
                current_batch->batch_index = batch_index;
                current_batch->batch_index_right = batch_index_right;
            }
        }
        else
        {
            batch_index++;
        }
        current_batch->batch_index = batch_index;
        std::cout << "Batch " << batch_index << "finished" << ":\n";
    }

    if (is_aggregate)
    {
        aggregate_op->intermidiate_results->total_rows = total_rows;
        aggregate_op->finalizeAggregates(*aggregate_op->intermidiate_results);
        aggregate_op->intermidiate_results->write_to_file();
        // aggregate_op->intermidiate_results->print();
    }
    else if (is_order_by)
    {
        TableResults final_result = order_by_op->mergeSortedBatchesOnGPU(order_by_batches);
        final_result.write_to_file();
        // final_result.print();
    }

    delete current_batch;
}