#include "physical_plan/order_by.hpp"

OrderBy::OrderBy(const duckdb::InsertionOrderPreservingMap<std::string> &params) : PhysicalOpNode()
{
    auto it = params.find("__order_by__");
    if (it != params.end())
    {
        // memory.main.table_1.id
        std::string text = it->second;
        text = text.substr(12);
        size_t pos = text.find('.');
        size_t pos2 = text.find(' ', pos);
        if (pos != std::string::npos && pos2 != std::string::npos)
        {
            table_name = text.substr(0, pos);
            col_name = text.substr(pos + 1, pos2 - pos);
            std::string order = text.substr(pos2 + 1);
            if (order == "DESC")
            {
                is_Ascending = false;
            }
            else
            {
                is_Ascending = true;
            }
        }
        std::cout << "table_name: " << table_name << std::endl;
        std::cout << "col_name: " << col_name << std::endl;
        std::cout << "order: " << is_Ascending << std::endl;
    }
}

void OrderBy::print() const
{
    // std::cout << "OrderBy: " << col_name << " " << order;
}