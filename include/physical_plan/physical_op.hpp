#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <future>
#include "duckdb.hpp"
#include "constants/db.hpp"
#include "dbms/csv_importer.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
    static void checkCudaError(cudaError_t err, const char *msg, const char *file, int line)
    {
        if (err != cudaSuccess)
        {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err) << " at " << file << ":" << line;
            throw std::runtime_error(ss.str());
        }
    }
}
class PhysicalOpNode
{
public:
    duckdb::InsertionOrderPreservingMap<std::string> params; // Operator parameters
    std::vector<std::unique_ptr<PhysicalOpNode>> children;   // Child nodes

    virtual ~PhysicalOpNode() = default;

    // Pure virtual print function to be overridden by derived classes
    virtual void print() const = 0;

    // Static method to build the tree and return the root node
    static std::unique_ptr<PhysicalOpNode> buildPlanTree(duckdb::PhysicalOperator *op, DB *data_base, TableResults **input_table_ptr, size_t batch_index, size_t batch_size, size_t batch_index_right, bool *is_join, bool *end_right);
    static void executePlanInBatches(duckdb::PhysicalOperator *op, DB *data_base, size_t batch_size);

protected:
    PhysicalOpNode() = default; // Protected default constructor for derived classes
};