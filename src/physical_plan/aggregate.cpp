// #include "physical_plan/aggregate.hpp"
// #include <variant>
// #include <sstream>
// #include <limits>
// Aggregate::Aggregate(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
// {
//     auto it = params.find("Aggregates");
//     if (it != params.end())
//     {
//         parseAggregateList(it->second);
//     }
// }

// void Aggregate::parseAggregateList(const std::string &aggregate_list)
// {
//     std::istringstream iss(aggregate_list);
//     std::string agg_str;

//     while (std::getline(iss, agg_str, '\n'))
//     {
//         agg_str.erase(0, agg_str.find_first_not_of(" \t"));
//         agg_str.erase(agg_str.find_last_not_of(" \t") + 1);

//         if (!agg_str.empty())
//         {
//             aggregates.push_back(parseSingleAggregate(agg_str));
//         }
//     }
// }

// AggregateFunction Aggregate::parseSingleAggregate(const std::string &agg_str) const
// {
//     std::istringstream iss(agg_str);
//     std::string func_name;
//     iss >> func_name;

//     AggregateFunction func;

//     if (func_name == "count_star()")
//     {
//         func.type = AggregateType::COUNT_STAR;
//         func.column_index = -1;
//     }
//     else if (func_name.find("count(#") == 0)
//     {
//         func.type = AggregateType::COUNT;
//         func.column_index = std::stoi(agg_str.substr(7, agg_str.find(')') - 7));
//     }
//     else if (func_name.find("avg(#") == 0)
//     {
//         func.type = AggregateType::AVG;
//         func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
//     }
//     else if (func_name.find("min(#") == 0)
//     {
//         func.type = AggregateType::MIN;
//         func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
//     }
//     else if (func_name.find("max(#") == 0)
//     {
//         func.type = AggregateType::MAX;
//         func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
//     }
//     else if (func_name.find("sum(#") == 0)
//     {
//         func.type = AggregateType::SUM;
//         func.column_index = std::stoi(agg_str.substr(5, agg_str.find(')') - 5));
//     }
//     else
//     {
//         throw std::runtime_error("Unknown aggregate function: " + agg_str);
//     }

//     return func;
// }

// TableResults Aggregate::computeAggregates(const TableResults &input) const
// {
//     TableResults result;
//     result.row_count = 1;
//     result.column_count = aggregates.size();

//     // Initialize result columns
//     for (size_t i = 0; i < aggregates.size(); ++i)
//     {
//         ColumnInfo col;
//         col.name = getAggregateName(aggregates[i], input);
//         col.type = getOutputType(aggregates[i], input);
//         result.columns.push_back(col);
//     }

//     // Process each aggregate
//     for (size_t i = 0; i < aggregates.size(); ++i)
//     {
//         const auto &agg = aggregates[i];

//         switch (agg.type)
//         {
//         case AggregateType::COUNT_STAR:
//             result.rows.push_back(static_cast<int64_t>(input.row_count));
//             break;

//         case AggregateType::COUNT:
//             result.rows.push_back(static_cast<int64_t>(countNonNull(input, agg.column_index)));
//             break;

//         case AggregateType::SUM:
//             result.rows.push_back(computeSum(input, agg.column_index));
//             break;

//         case AggregateType::AVG:
//             result.rows.push_back(computeAvg(input, agg.column_index));
//             break;

//         case AggregateType::MIN:
//             result.rows.push_back(findMin(input, agg.column_index));
//             break;

//         case AggregateType::MAX:
//             result.rows.push_back(findMax(input, agg.column_index));
//             break;
//         }
//     }

//     return result;
// }

// // Helper function to check for null values
// bool Aggregate::isNull(const ValueVariant &val) const
// {
//     return val.valueless_by_exception();
// }

// std::string Aggregate::getAggregateName(const AggregateFunction &agg, const TableResults &input) const
// {
//     switch (agg.type)
//     {
//     case AggregateType::COUNT_STAR:
//         return "count_star";
//     case AggregateType::COUNT:
//         return "count_" + input.columns[agg.column_index].name;
//     case AggregateType::SUM:
//         return "sum_" + input.columns[agg.column_index].name;
//     case AggregateType::AVG:
//         return "avg_" + input.columns[agg.column_index].name;
//     case AggregateType::MIN:
//         return "min_" + input.columns[agg.column_index].name;
//     case AggregateType::MAX:
//         return "max_" + input.columns[agg.column_index].name;
//     }
//     return "agg";
// }

// DataType Aggregate::getOutputType(const AggregateFunction &agg, const TableResults &input) const
// {
//     switch (agg.type)
//     {
//     case AggregateType::COUNT_STAR:
//     case AggregateType::COUNT:
//         return DataType::FLOAT;
//     case AggregateType::SUM:
//     case AggregateType::AVG:
//         return DataType::FLOAT;
//     case AggregateType::MIN:
//     case AggregateType::MAX:
//         return input.columns[agg.column_index].type;
//     }
//     return DataType::INT;
// }

// int64_t Aggregate::countNonNull(const TableResults &input, int col_idx) const
// {
//     int64_t count = 0;
//     for (size_t row = 0; row < input.row_count; ++row)
//     {
//         size_t idx = row * input.column_count + col_idx;
//         if (idx < input.rows.size() && !isNull(input.rows[idx]))
//         {
//             count++;
//         }
//     }
//     return count;
// }

// float Aggregate::computeSum(const TableResults &input, int col_idx) const
// {
//     double sum = 0;
//     for (size_t row = 0; row < input.row_count; ++row)
//     {
//         size_t idx = row * input.column_count + col_idx;
//         if (idx < input.rows.size() && !isNull(input.rows[idx]))
//         {
//             if (std::holds_alternative<float>(input.rows[idx]))
//             {
//                 sum += std::get<float>(input.rows[idx]);
//             }
//             else if (std::holds_alternative<int>(input.rows[idx]))
//             {
//                 sum += std::get<int>(input.rows[idx]);
//             }
//             else if (std::holds_alternative<int64_t>(input.rows[idx]))
//             {
//                 sum += std::get<int64_t>(input.rows[idx]);
//             }
//         }
//     }
//     return static_cast<float>(sum);
// }

// float Aggregate::computeAvg(const TableResults &input, int col_idx) const
// {
//     double sum = 0;
//     int64_t count = 0;
//     for (size_t row = 0; row < input.row_count; ++row)
//     {
//         size_t idx = row * input.column_count + col_idx;
//         if (idx < input.rows.size() && !isNull(input.rows[idx]))
//         {
//             if (std::holds_alternative<float>(input.rows[idx]))
//             {
//                 sum += std::get<float>(input.rows[idx]);
//                 count++;
//             }
//             else if (std::holds_alternative<int>(input.rows[idx]))
//             {
//                 sum += std::get<int>(input.rows[idx]);
//                 count++;
//             }
//             else if (std::holds_alternative<int64_t>(input.rows[idx]))
//             {
//                 sum += std::get<int64_t>(input.rows[idx]);
//                 count++;
//             }
//         }
//     }
//     return count > 0 ? static_cast<float>(sum / count) : 0.0f;
// }

// ValueVariant Aggregate::findMin(const TableResults &input, int col_idx) const
// {
//     ValueVariant min_val;
//     bool has_value = false;

//     for (size_t row = 0; row < input.row_count; ++row)
//     {
//         size_t idx = row * input.column_count + col_idx;
//         if (idx >= input.rows.size() || isNull(input.rows[idx]))
//             continue;

//         const auto &val = input.rows[idx];
//         if (!has_value)
//         {
//             min_val = val;
//             has_value = true;
//         }
//         else
//         {
//             if (val < min_val)
//             {
//                 min_val = val;
//             }
//         }
//     }

//     if (!has_value)
//     {
//         // Return default value based on column type
//         DataType type = input.columns[col_idx].type;
//         switch (type)
//         {
//         case DataType::FLOAT:
//             return 0.0f;
//         case DataType::INT:
//             return 0;
//         case DataType::STRING:
//             return std::string();
//         default:
//             return 0.0f;
//         }
//     }
//     return min_val;
// }

// ValueVariant Aggregate::findMax(const TableResults &input, int col_idx) const
// {
//     ValueVariant max_val;
//     bool has_value = false;

//     for (size_t row = 0; row < input.row_count; ++row)
//     {
//         size_t idx = row * input.column_count + col_idx;
//         if (idx >= input.rows.size() || isNull(input.rows[idx]))
//             continue;

//         const auto &val = input.rows[idx];
//         if (!has_value)
//         {
//             max_val = val;
//             has_value = true;
//         }
//         else
//         {
//             if (val > max_val)
//             {
//                 max_val = val;
//             }
//         }
//     }

//     if (!has_value)
//     {
//         // Return default value based on column type
//         DataType type = input.columns[col_idx].type;
//         switch (type)
//         {
//         case DataType::FLOAT:
//             return 0.0f;
//         case DataType::INT:
//             return 0;
//         case DataType::STRING:
//             return std::string();
//         default:
//             return 0.0f;
//         }
//     }
//     return max_val;
// }
// void Aggregate::print() const
// {
//     std::cout << "UNGROUPED_AGGREGATE (";
//     for (size_t i = 0; i < aggregates.size(); ++i)
//     {
//         if (i > 0)
//             std::cout << ", ";

//         switch (aggregates[i].type)
//         {
//         case AggregateType::COUNT_STAR:
//             std::cout << "count_star()";
//             break;
//         case AggregateType::COUNT:
//             std::cout << "count(#" << aggregates[i].column_index << ")";
//             break;
//         case AggregateType::SUM:
//             std::cout << "sum(#" << aggregates[i].column_index << ")";
//             break;
//         case AggregateType::AVG:
//             std::cout << "avg(#" << aggregates[i].column_index << ")";
//             break;
//         case AggregateType::MIN:
//             std::cout << "min(#" << aggregates[i].column_index << ")";
//             break;
//         case AggregateType::MAX:
//             std::cout << "max(#" << aggregates[i].column_index << ")";
//             break;
//         }

//         if (!output_names.empty() && i < output_names.size())
//         {
//             std::cout << " AS " << output_names[i];
//         }
//     }
//     std::cout << ")\n";
// }