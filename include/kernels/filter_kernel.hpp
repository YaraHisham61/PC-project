#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"

template <typename T>
__global__ void filterKernel(
    T *data_in,
    bool *output_mask,
    size_t row_count,
    int threshold,
    uint8_t cond
);
