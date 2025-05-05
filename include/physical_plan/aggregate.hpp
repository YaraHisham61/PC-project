#pragma once
#include "physical_plan/physical_op.hpp"
#include "kernels/aggregate/max_kernel.hpp"
#include "kernels/aggregate/min_kernel.hpp"
#include "kernels/aggregate/sum_kernel.hpp"
#include "kernels/aggregate/count_kernel.hpp"
#include "kernels/aggregate/count_star_kernel.hpp"
#include <cuda_runtime.h>
#include <sstream>

enum class AggregateType
{
    COUNT_STAR,
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX
};

struct AggregateFunction
{
    AggregateType type;
    int column_index;
};

class Aggregate : public PhysicalOpNode
{
public:
    std::vector<AggregateFunction> aggregates;
    std::vector<std::string> output_names;
    TableResults *intermidiate_results = nullptr;

    Aggregate(const duckdb::InsertionOrderPreservingMap<std::string> &params);
    TableResults computeAggregates(const TableResults &input) const;
    void print() const override;
    ~Aggregate() = default;
    std::string getAggregateName(const AggregateFunction &agg, const TableResults &input) const;
    DataType getOutputType(const AggregateFunction &agg, const TableResults &input) const;
    uint64_t countNonNull(const TableResults &input, int col_idx) const;
    void updateAggregates(const TableResults &input);
        // float computeSum(const TableResults &input, int col_idx) const;
        // float computeAvg(const TableResults &input, int col_idx) const;
        // ValueVariant findMin(const TableResults &input, int col_idx) const;
        // ValueVariant findMax(const TableResults &input, int col_idx) const;
        void
        parseAggregateList(const std::string &aggregate_list);
    // bool isNull(const ValueVariant &val) const;
    AggregateFunction parseSingleAggregate(const std::string &agg_str) const;
};