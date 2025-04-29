// #pragma once

// #include "physical_plan/physical_op.hpp"
// #include <regex>
// #include <cctype>
// struct Condition {
//     std::string column;
//     std::string op;
//     std::string value;
//     bool is_string;
// };

// class Filter : public PhysicalOpNode {
//     std::vector<Condition> conditions;
//     std::vector<std::string> logical_ops;

// public:
//     Filter(const duckdb::InsertionOrderPreservingMap<std::string>& params);
//     TableResults applyFilter(const TableResults& input_table) const;
//     void print() const override;
//     std::string trim(const std::string& str) const;
//     void parseConditions(const std::string& expression);
//     void parseSimpleCondition(const std::string& expr);
//     void parseComplexCondition(const std::string& expr);
//     Condition parseSingleCondition(const std::string& cond_expr) const;
//     bool evaluateRow(const TableResults& table, size_t row_idx, size_t num_cols) const;
//     bool evaluateCondition(const TableResults& table, size_t row_idx, size_t num_cols, const Condition& cond) const;
// };
