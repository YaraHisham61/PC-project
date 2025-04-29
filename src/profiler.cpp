#include "dbms/profiler.hpp"

void Profiler::start(const std::string &name)
{
    starts[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::stop(const std::string &name)
{
    auto end = std::chrono::high_resolution_clock::now();
    times[name] = std::chrono::duration_cast<std::chrono::milliseconds>(end - starts[name]).count();
    std::cout << name << " Time: " << times[name] << " ms\n";
}

void Profiler::saveResults(const std::string &file)
{
    std::ofstream out(file);
    out << "Operation,Time_ms\n";
    for (const auto &[name, time] : times)
    {
        out << name << "," << time << "\n";
    }
    out.close();
}