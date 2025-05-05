#include "duckdb.hpp"
#include "dbms/csv_importer.hpp"
#include "dbms/duckdb_interface.hpp"
#include "dbms/profiler.hpp"
#include "constants/db.hpp"
#include "physical_plan/physical_op.hpp"
#include "duckdb/execution/executor.hpp"
#include <memory>
void printPhysicalPlan(duckdb::PhysicalOperator *node, int depth = 0)
{
    if (!node)
        return;

    std::string op_name = node->GetName();
    auto params = node->ParamsToString();
    op_name.erase(std::remove(op_name.begin(), op_name.end(), ' '), op_name.end());

    for (auto &child : node->children)
    {
        printPhysicalPlan(&(child.get()), depth + 1);
    }
    // Print the current node with indentation
    std::cout << op_name << std::endl;
    for (auto param : params)
    {

        std::cout << param.first << ": " << param.second << std::endl;
    }
}
int main()
{
    Profiler profiler;
    profiler.start("Total");
    // Create in-memory database
    DB data_base;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    DuckDBInterface duckdb_interface(db, con);

    con.Query("SET disabled_optimizers = 'filter_pushdown, statistics_propagation';"); // Disable returning empty results
    // Create sample data

    profiler.start("Import CSV");
    CSVImporter csv_importer(db, con);
    bool x = csv_importer.import_folder(DATA_DIR, &data_base);

    if (!x)
    {
        std::cerr << "Error importing CSV file." << std::endl;
        return 1;
    }
    else
    {
        std::cout << "CSV file imported successfully." << std::endl;
    }
    profiler.stop("Import CSV");
    data_base.print_databse();
    con.BeginTransaction();
    // Query to analyze

    std::string query = "SELECT * FROM table1 order by id ASC;";
    // std::string query = "SELECT COUNT(name) FROM Student;";
    // std::string query = "SELECT * FROM Student WHERE year >2019 or id = 500; ";
    // std::string query = "SELECT UPPER(name),id AS name_upper FROM Student;";
    // std::string query = "SELECT count(*) ,count(name) FROM Student;";
    // std::string query = "SELECT t1.id , t5.table_1_id , t4.table_1_id FROM  table_1 t1 , table_5 t5, table_4 t4 where t1.id = t5.table_1_id and t1.id = t4.table_1_id and t1.id >8000";
    // std::string query = "SELECT * FROM  table1 t4 , table4 t1 WHERE t4.id= t1.table_1_id and t1.last_modified = t4.completion_date;";
    // std::string query = "SELECT * FROM  table1 t4 , table4 t1 WHERE t4.id= t1.table_1_id and t1.last_modified = t4.completion_date;";

    // std::string query = "SELECT id,year,name,name FROM Student;";
    profiler.start("Get Logical Plan");
    auto logical_plan = duckdb_interface.getLogicalPlan(query);
    profiler.stop("Get Logical Plan");
    std::cout << "Logical plan:\n"
              << logical_plan->ToString() << std::endl;

    profiler.start("Get Physical Plan");
    duckdb::PhysicalPlanGenerator physical_plan_generator(*con.context);
    auto physical_plan = physical_plan_generator.Plan(logical_plan->Copy(*con.context));
    profiler.stop("Get Physical Plan");

    std::cout << "Physical plan:\n";

    std::cout << physical_plan.get()->Root().ToString() << std::endl;

    // profiler.start("CPU Execution");
    // std::string csv_file = std::string(DATA_DIR) + "Student.csv";
    // std::string create_table_query = "CREATE TABLE Student AS SELECT * FROM '" + csv_file + "';";
    // auto create_result = con.Query(create_table_query);
    // auto result = con.Query(query);
    // profiler.stop("CPU Execution");
    // std::cout << "CPU Execution Result:\n";
    // if (result->HasError())
    // {
    //     std::cerr << "Query execution failed: " << result->GetError() << std::endl;
    //     return 1;
    // }
    // result->Print();

    // printPhysicalPlan(&(physical_plan.get()->Root()), 2);

    // PhysicalOpNode root_node;
    TableResults *input_table = nullptr;
    // auto plan_tree = PhysicalOpNode::buildPlanTree(&(physical_plan.get()->Root()), &data_base, &input_table);
    PhysicalOpNode::executePlanInBatches(&(physical_plan.get()->Root()), &data_base, 10);
    // printPhysicalPlan(&(physical_plan.get()->Root()));

    // TableResults r = csv_importer.read_table(&data_base, "Student", {"name", "year"});
    // r.print();
    // root_node.buildPlanTree(&(physical_plan.get()->Root()), &root_node);
    // std::cout << "Physsical plan tree:\n";
    // plan_tree->print();

    // if (status == cudaSuccess)
    // {
    //     // Process result (boolean mask of rows that passed filter)
    //     for (size_t i = 0; i < result.size(); ++i)
    //     {
    //         if (result[i])
    //         {
    //             // Row i passed the filter
    //             std::cout << "Row " << i << " passed the filter." << std::endl;
    //         }
    //     }
    // }
    // else
    // {
    //     // Handle CUDA error
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    // }

    // // printPhysicalPlan(&(physical_plan.get()->Root()), 2);

    profiler.stop("Total");

    return 0;
}