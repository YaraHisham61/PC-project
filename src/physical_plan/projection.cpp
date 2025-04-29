// #include "physical_plan/projection.hpp"

// Projection::Projection(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
// {
//     // Initialize the projection based on the provided parameters
//     auto it = params.find("__projections__");
//     if (it != params.end())
//     {
//         parseProjectionList(it->second);
//     }
// }

// void Projection::parseProjectionList(const std::string &projection_list)
// {
//     std::istringstream iss(projection_list);
//     std::string line;

//     flag = projection_list.find('#') != std::string::npos;

//     while (std::getline(iss, line))
//     {
//         line.erase(0, line.find_first_not_of(" \t\n\r"));
//         line.erase(line.find_last_not_of(" \t\n\r") + 1);

//         if (line.empty())
//             continue;
//         if (flag)
//         {
//             projections_index.push_back(line[1] - '0');
//         }
//         else
//         {
//             output_names.push_back(line);
//         }
//     }
// }

// TableResults Projection::applyProjection(const TableResults &input_table) const
// {
//     TableResults result;

//     // Handle empty input
//     if (input_table.rows.empty() || input_table.columns.empty())
//     {
//         return result;
//     }

//     // Get the number of columns and rows from input
//     size_t num_input_cols = input_table.columns.size();
//     size_t num_rows = input_table.row_count;

//     // Case 1: Index-based projection (like "#1\n#2")
//     if (flag)
//     {
//         // Set up the result columns
//         for (int index : projections_index)
//         {
//             // Validate index
//             if (index < 0 || index >= static_cast<int>(num_input_cols))
//             {
//                 throw std::runtime_error("Projection index out of bounds");
//             }
//             result.columns.push_back(input_table.columns[index]);
//         }

//         // Copy the data for selected columns
//         for (size_t row = 0; row < num_rows; ++row)
//         {
//             for (int col_index : projections_index)
//             {
//                 size_t flat_index = row * num_input_cols + col_index;
//                 if (flat_index < input_table.rows.size())
//                 {
//                     result.rows.push_back(input_table.rows[flat_index]);
//                 }
//             }
//         }

//         result.row_count = num_rows;
//         result.column_count = projections_index.size();
//     }
//     // Case 2: Column renaming only (like "id\nname_upper")
//     else
//     {
//         // Copy all data from input
//         result = input_table;
//     }

//     // Update column names with output_names if provided
//     if (!output_names.empty())
//     {
//         // Validate output names count matches result columns count
//         if (output_names.size() != result.columns.size())
//         {
//             throw std::runtime_error("Output names count doesn't match projection result columns count");
//         }

//         // Apply new names
//         for (size_t i = 0; i < result.columns.size(); ++i)
//         {
//             result.columns[i].name = output_names[i];
//         }
//     }

//     return result;
// }
// void Projection::print() const
// {
//     std::cout << "Projection: ";
//     for (const auto &name : output_names)
//     {
//         std::cout << name << " ";
//     }
//     std::cout << "\n";
// }