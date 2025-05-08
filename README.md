# GPU-Accelerated Database Management System

This project implements a GPU-accelerated database management system that processes SQL queries using both CPU and GPU execution paths. The system uses DuckDB as the query optimizer and planner, while implementing custom GPU kernels for query execution.

## Prerequisites

- CUDA Toolkit (version 11.0 or higher)
- CMake (version 3.10 or higher)
- C++17 compatible compiler
- WSL2 (if running on Windows)

## Building DuckDB

1. Clone DuckDB repository:
```bash
git clone https://github.com/duckdb/duckdb.git
cd duckdb
```

2. Build DuckDB:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## Project Structure

```
dbms/
├── data/               # Contains CSV data files and query files
├── include/           # Header files
│   ├── constants/     # Database constants
│   ├── csv_parser/    # CSV parsing utilities
│   ├── dbms/          # Core DBMS components
│   ├── kernels/       # CUDA kernels
│   └── physical_plan/ # Physical plan execution
├── src/               # Source files
└── CMakeLists.txt     # Build configuration
```

## Building the Project

1. Create a build directory and navigate to it:
```bash
mkdir build
cd build
```

2. Configure and build the project:
```bash
cmake ..
make
```

## Running the Project

The project requires two command-line arguments:
1. Path to the data folder containing CSV files
2. Path to the query file

Example usage:
```bash
./build/PC-project /path/to/data/folder /path/to/query.txt
```

For example:
```bash
./build/PC-project data data/query2.txt
```

The program will:
1. Import CSV files from the specified data folder
2. Execute the query from the specified query file
3. Run both CPU and GPU execution paths
4. Display performance metrics using the built-in profiler
5. Generate output files in the current directory:
   - `Team9_<query_filename>.csv`: GPU execution results

For example, if you run with `query1.txt`, the following files will be created:
- `Team9_query1.csv`

## Data Format

The system expects CSV files in the following format:
- Each table should have a corresponding CSV file in the data directory
- CSV files should be named according to their table names (e.g., `table_1.csv`)
- The first row should contain column names
- Data should be comma-separated

## Query Format

Queries should be written in standard SQL format and saved in a text file. The system supports:
- SELECT statements
- JOIN operations
- WHERE clauses
- Basic aggregations

Example query file (`query1.txt`):
```sql
SELECT * FROM table_1 JOIN table_4 ON table_1.id = table_4.id WHERE table_1.value > 100;
```

## Performance Profiling

The system includes a built-in profiler that measures:
- Total execution time
- CSV import time
- Logical plan generation time
- Physical plan generation time
- CPU execution time
- GPU execution time

The profiler output will be displayed after each run.