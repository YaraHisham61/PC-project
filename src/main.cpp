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

    // con.Query("PRAGMA enable_profiling");
    // con.Query("PRAGMA profiling_mode=detailed");

    con.Query("SET disabled_optimizers = 'filter_pushdown, statistics_propagation';"); // Disable returning empty results
    con.Query("SET threads TO 1;");

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
    // data_base.print_databse();
    con.BeginTransaction();
    // Query to analyze

    // std::string query = "SELECT * FROM table1 order by id ASC;";
    // std::string query = "SELECT COUNT(name) FROM Student;";
    // id(N)(P), completion_date(D), longitude(N), total(N)

    // std::string query = "SELECT id FROM table_1 ORDER BY id; ";
    std::string query = "SELECT AVG(e.Salary) AS AverageSalary FROM Employees e,SalesOrders s WHERE e.Employees_id = s.Employees_id AND s.TotalAmount > 200; ";

    // std::string query = "SELECT UPPER(name),id AS name_upper FROM Student;";
    // std::string query = "SELECT count(*) ,count(name) FROM Student;";
    // std::string query = "SELECT t1.id , t5.table_1_id , t4.table_1_id FROM  table_1 t1 , table_5 t5, table_4 t4 where t1.id = t5.table_1_id and t1.id = t4.table_1_id and t1.id >8000";
    // std::string query = "SELECT * FROM  table1 t4 , table4 t1 WHERE t4.id= t1.table_1_id and t1.last_modified = t4.completion_date;";
    // std::string query = "SELECT * FROM  table1 t4 , table4 t1 WHERE t4.id= t1.table_1_id and t1.last_modified = t4.completion_date;";

    // std::string query = "SELECT id,year,name,name FROM Student;";
    profiler.start("Get Logical Plan");
    auto logical_plan = duckdb_interface.getLogicalPlan(query);
    profiler.stop("Get Logical Plan");
    // std::cout << "Logical plan:\n"
    //           << logical_plan->ToString() << std::endl;

    profiler.start("Get Physical Plan");
    duckdb::PhysicalPlanGenerator physical_plan_generator(*con.context);
    auto physical_plan = physical_plan_generator.Plan(logical_plan->Copy(*con.context));
    profiler.stop("Get Physical Plan");

    // std::cout << "Physical plan:\n";

    std::cout << physical_plan.get()->Root().ToString() << std::endl;

    // profiler.start("CPU Execution");
    // std::string csv_file = std::string(DATA_DIR) + "table_1.csv";
    // // std::string create_table_query = "INSERT INTO Student SELECT * FROM read_csv('" + csv_file + "',header=false);";
    // std::string create_table_query = "COPY table_1 FROM '" + csv_file + "';";
    // auto create_result = con.Query(create_table_query);
    // auto result = con.Query(query);
    // profiler.stop("CPU Execution");
    // // std::cout << "CPU Execution Result:\n";
    // if (result->HasError())
    // {
    //     std::cerr << "Query execution failed: " << result->GetError() << std::endl;
    //     return 1;
    // }
    // // result->Print();

    profiler.start("GPU Execution");
    PhysicalOpNode::executePlanInBatches(&(physical_plan.get()->Root()), &data_base, 1000000);
    // printPhysicalPlan(&(physical_plan.get()->Root()));
    profiler.stop("GPU Execution");

    profiler.stop("Total");

    return 0;
}