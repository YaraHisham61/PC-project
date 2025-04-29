// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <unordered_map>
// #include <memory>
// #include <functional>
// #include <algorithm>
// #include <variant>
// #include <any>

// // Forward declarations
// class PhysicalOperator;
// class TableScanOperator;
// class FilterOperator;
// class ProjectionOperator;
// class HashGroupByOperator;

// // Type aliases for flexibility
// using DataRow = std::unordered_map<std::string, std::any>;
// using DataTable = std::vector<DataRow>;
// using StringVector = std::vector<std::string>;

// // Base class for all physical operators
// class PhysicalOperator
// {
// public:
//     virtual ~PhysicalOperator() = default;
//     virtual DataTable execute() = 0;
//     virtual void setSource(std::shared_ptr<PhysicalOperator> source)
//     {
//         source_ = source;
//     }

// protected:
//     std::shared_ptr<PhysicalOperator> source_;
// };

// // Table Scan Operator
// class TableScanOperator : public PhysicalOperator
// {
// public:
//     TableScanOperator(const std::string &table_name, const StringVector &projections)
//         : table_name_(table_name), projections_(projections) {}

//     DataTable execute() override
//     {
//         DataTable result;
//         std::ifstream file(table_name_ + ".csv");
//         std::string line;

//         // Read header to get column names
//         std::getline(file, line);
//         auto headers = splitCSVLine(line);

//         while (std::getline(file, line))
//         {
//             auto values = splitCSVLine(line);
//             DataRow row;

//             for (size_t i = 0; i < headers.size(); ++i)
//             {
//                 // Only include columns in projections if specified
//                 if (projections_.empty() ||
//                     std::find(projections_.begin(), projections_.end(), headers[i]) != projections_.end())
//                 {
//                     row[headers[i]] = values[i];
//                 }
//             }
//             result.push_back(row);
//         }

//         return result;
//     }

// private:
//     std::string table_name_;
//     StringVector projections_;

//     std::vector<std::string> splitCSVLine(const std::string &line)
//     {
//         std::vector<std::string> result;
//         size_t start = 0, end = line.find(',');

//         while (end != std::string::npos)
//         {
//             result.push_back(line.substr(start, end - start));
//             start = end + 1;
//             end = line.find(',', start);
//         }
//         result.push_back(line.substr(start));

//         return result;
//     }
// };

// // Filter Operator
// class FilterOperator : public PhysicalOperator
// {
// public:
//     using FilterPredicate = std::function<bool(const DataRow &)>;

//     FilterOperator(FilterPredicate predicate) : predicate_(predicate) {}

//     DataTable execute() override
//     {
//         DataTable input = source_->execute();
//         DataTable result;

//         std::copy_if(input.begin(), input.end(), std::back_inserter(result),
//                      [this](const DataRow &row)
//                      { return predicate_(row); });

//         return result;
//     }

// private:
//     FilterPredicate predicate_;
// };

// // Projection Operator
// class ProjectionOperator : public PhysicalOperator
// {
// public:
//     ProjectionOperator(const StringVector &projections) : projections_(projections) {}

//     DataTable execute() override
//     {
//         DataTable input = source_->execute();
//         DataTable result;

//         for (const auto &row : input)
//         {
//             DataRow new_row;
//             for (const auto &col : projections_)
//             {
//                 if (row.find(col) != row.end())
//                 {
//                     new_row[col] = row.at(col);
//                 }
//             }
//             result.push_back(new_row);
//         }

//         return result;
//     }

// private:
//     StringVector projections_;
// };

// // Hash Group By Operator
// class HashGroupByOperator : public PhysicalOperator
// {
// public:
//     using AggregateFunction = std::function<void(DataRow &, const DataRow &)>;

//     HashGroupByOperator(const StringVector &group_keys,
//                         const std::unordered_map<std::string, AggregateFunction> &aggregates)
//         : group_keys_(group_keys), aggregates_(aggregates) {}

//     DataTable execute() override
//     {
//         DataTable input = source_->execute();
//         std::unordered_map<std::string, DataRow> groups;

//         for (const auto &row : input)
//         {
//             // Create group key
//             std::string key;
//             for (const auto &k : group_keys_)
//             {
//                 if (row.find(k) != row.end())
//                 {
//                     try
//                     {
//                         key += std::any_cast<std::string>(row.at(k)) + "|";
//                     }
//                     catch (...)
//                     {
//                         // Handle other types if needed
//                     }
//                 }
//             }

//             // Initialize group if not exists
//             if (groups.find(key) == groups.end())
//             {
//                 groups[key] = DataRow();
//                 for (const auto &k : group_keys_)
//                 {
//                     if (row.find(k) != row.end())
//                     {
//                         groups[key][k] = row.at(k);
//                     }
//                 }
//                 // Initialize aggregates
//                 for (const auto &agg : aggregates_)
//                 {
//                     groups[key][agg.first] = nullptr; // Initialize as appropriate
//                 }
//             }

//             // Apply aggregates
//             for (const auto &agg : aggregates_)
//             {
//                 agg.second(groups[key], row);
//             }
//         }

//         // Convert map to vector
//         DataTable result;
//         for (const auto &pair : groups)
//         {
//             result.push_back(pair.second);
//         }

//         return result;
//     }

// private:
//     StringVector group_keys_;
//     std::unordered_map<std::string, AggregateFunction> aggregates_;
// };

// // Physical Plan Builder
// class PhysicalPlanBuilder
// {
// public:
//     std::shared_ptr<PhysicalOperator> build(const std::string &plan_description)
//     {
//         // In a real implementation, you would parse the plan_description
//         // and build the operator tree accordingly

//         // For demonstration, we'll build the plan from your example:
//         // PROJECTION (name)
//         //   HASH_GROUP_BY (group by name)
//         //     PROJECTION (year, name)
//         //       FILTER (year > 2019)
//         //         SEQ_SCAN (Student, projections: year, name)

//         // Create operators
//         auto scan = std::make_shared<TableScanOperator>("Student", StringVector{"year", "name"});

//         auto filter = std::make_shared<FilterOperator>(
//             [](const DataRow &row)
//             {
//                 try
//                 {
//                     int year = std::stoi(std::any_cast<std::string>(row.at("year")));
//                     return year > 2019;
//                 }
//                 catch (...)
//                 {
//                     return false;
//                 }
//             });
//         filter->setSource(scan);

//         auto proj1 = std::make_shared<ProjectionOperator>(StringVector{"year", "name"});
//         proj1->setSource(filter);

//         // Simple count aggregate for demonstration
//         auto group_by = std::make_shared<HashGroupByOperator>(
//             StringVector{"name"},
//             std::unordered_map<std::string, HashGroupByOperator::AggregateFunction>{
//                 {"count", [](DataRow &group, const DataRow &row)
//                  {
//                      if (group.find("count") == group.end())
//                      {
//                          group["count"] = 1;
//                      }
//                      else
//                      {
//                          group["count"] = std::any_cast<int>(group["count"]) + 1;
//                      }
//                  }}});
//         group_by->setSource(proj1);

//         auto proj2 = std::make_shared<ProjectionOperator>(StringVector{"name"});
//         proj2->setSource(group_by);

//         return proj2;
//     }
// };

// int main()
// {
//     // Build and execute the physical plan
//     PhysicalPlanBuilder builder;
//     auto plan = builder.build("..."); // Plan description would be parsed here

//     DataTable result = plan->execute();

//     // Print results
//     std::cout << "Query Results:\n";
//     for (const auto &row : result)
//     {
//         for (const auto &col : row)
//         {
//             try
//             {
//                 std::cout << col.first << ": " << std::any_cast<std::string>(col.second) << "\t";
//             }
//             catch (...)
//             {
//                 try
//                 {
//                     std::cout << col.first << ": " << std::any_cast<int>(col.second) << "\t";
//                 }
//                 catch (...)
//                 {
//                     std::cout << col.first << ": [unprintable]\t";
//                 }
//             }
//         }
//         std::cout << "\n";
//     }

//     return 0;
// }