#pragma once

#include <chrono>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

class Profiler
{
    std::map<std::string, std::chrono::high_resolution_clock::time_point> starts;
    std::map<std::string, double> times;

public:
    void start(const std::string &name);
    void stop(const std::string &name);
    void saveResults(const std::string &file);
};
