#pragma once
#include <ctime>
#include <cstdint>
#include <string>

#define MAX_STR_LEN 20

enum class DataType : __uint8_t
{
    FLOAT,
    DATETIME,
    STRING
};

[[nodiscard]] inline std::string getDataTypeString(DataType type)
{
    switch (type)
    {
    case DataType::FLOAT:
        return "FLOAT";
    case DataType::DATETIME:
        return "DATETIME";
    case DataType::STRING:
        return "STRING";
    default:
        return "FLOAT";
    }
}

[[nodiscard]] inline uint8_t getDataTypeNumBytes(DataType type)
{

    if (type == DataType::FLOAT)
        return sizeof(float); // 32-bit
    if (type == DataType::DATETIME)
        return sizeof(uint64_t); // 64-bit timestamp //TODO: recheck it
    if (type == DataType::STRING)
        return MAX_STR_LEN; // 1 char = 1 byte
    return 0;               // Fallback for unhandled cases
}
[[nodiscard]] inline uint64_t getDateTime(const std::string &valueStr)
{
    std::tm valueTm = {};
    // Initialize all fields to avoid garbage values
    valueTm.tm_hour = 0;
    valueTm.tm_min = 0;
    valueTm.tm_sec = 0;
    valueTm.tm_year = 0;
    valueTm.tm_mon = 0;
    valueTm.tm_mday = 0;

    // Parse the time string
    if (strptime(valueStr.c_str(), "%Y-%m-%d %H:%M:%S", &valueTm) == nullptr)
    {
        // Handle parsing error
        return 0;
    }

    // Convert to time_t (seconds since epoch)
    time_t timeValue = timegm(&valueTm); // Use timegm for UTC or mktime for local time

    return ((valueTm.tm_year + 1900) * 10000000000ULL) +
           ((valueTm.tm_mon + 1) * 100000000ULL) +
           (valueTm.tm_mday * 1000000ULL) +
           (valueTm.tm_hour * 10000ULL) +
           (valueTm.tm_min * 100ULL) +
           valueTm.tm_sec;
}

[[nodiscard]] inline std::string getDateTimeStr(uint64_t value)
{
    // Extract components from the custom uint64_t format
    int year = value / 10000000000ULL; // Extract YYYY
    value %= 10000000000ULL;
    int month = value / 100000000ULL; // Extract MM
    value %= 100000000ULL;
    int day = value / 1000000ULL; // Extract DD
    value %= 1000000ULL;
    int hour = value / 10000ULL; // Extract HH
    value %= 10000ULL;
    int minute = value / 100ULL; // Extract MM
    int second = value % 100ULL; // Extract SS

    // Create a std::tm structure
    std::tm tm_info = {};
    tm_info.tm_year = year - 1900; // tm_year is years since 1900
    tm_info.tm_mon = month - 1;    // tm_mon is 0-based (0-11)
    tm_info.tm_mday = day;
    tm_info.tm_hour = hour;
    tm_info.tm_min = minute;
    tm_info.tm_sec = second;

    // Format the time
    char buffer[26];
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", &tm_info);

    return std::string(buffer);
}