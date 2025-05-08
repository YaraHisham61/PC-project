#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

template <typename T>
__device__ bool operators(const T &a, const T &b, bool ascending);

template <typename T>
__device__ void merge(T *keys, size_t *indices, size_t *indicesTmp,
                      int left, int mid, int right, bool ascending);

template <typename T>
__global__ void mergeSortKernel(T *keys, size_t *indices, size_t *indicesTmp,
                                int n, int width, bool ascending);