cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(PC-project LANGUAGES CXX CUDA)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
# # Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# # Set build type
set(CMAKE_CUDA_ARCHITECTURES 75) 
# # use separate compiler for cuda files only
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(    
    include
    vendor/duckdb/src/include
)
link_directories(
    vendor/duckdb/build/src
)

# Collect all .cpp files in the src directory
file(GLOB_RECURSE CPP_FILES src/*.cpp)
file(GLOB_RECURSE CUDA_FILES src/*.cu)

add_executable(PC-project ${CPP_FILES} ${CUDA_FILES})
target_link_libraries(PC-project PRIVATE
    duckdb
    CUDA::cudart
    rt       # For librt
    pthread  # For libpthread
    dl       # For libdl
)
set_target_properties(PC-project PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CXX
)