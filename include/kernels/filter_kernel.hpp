#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <variant>
#include "constants/db.hpp"

// template <typename T>
__global__ void filterKernel(void *input, bool *output, size_t row_count, float value, uint8_t cond);
