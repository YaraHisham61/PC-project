#include "kernels/order_by.hpp"

template <typename T>
__device__ bool operators(const T &a, const T &b, bool ascending)
{
    return ascending ? (a < b) : (a > b);
}

template <>
__device__ bool operators<char *>(char *const &a, char *const &b, bool ascending)
{
    int cmp = 0;
    int i = 0;
    while (a[i] != '\0' && b[i] != '\0')
    {
        if (a[i] != b[i])
        {
            cmp = a[i] - b[i];
            break;
        }
        i++;
    }
    if (cmp == 0)
    {
        if (a[i] == '\0' && b[i] != '\0')
            cmp = -1;
        else if (a[i] != '\0' && b[i] == '\0')
            cmp = 1;
    }
    return ascending ? (cmp < 0) : (cmp > 0);
}

template <typename T>
__device__ void merge(T *keys, size_t *indices, size_t *indicesTmp,
                      int left, int mid, int right, bool ascending)
{
    int i = left;   
    int j = mid + 1; 
    int k = left;   

    while (i <= mid && j <= right)
    {
        if (operators<T>(keys[indices[i]], keys[indices[j]], ascending))
        {
            indicesTmp[k] = indices[i];
            i++;
        }
        else
        {
            indicesTmp[k] = indices[j];
            j++;
        }
        k++;
    }

    while (i <= mid)
    {
        indicesTmp[k] = indices[i];
        i++;
        k++;
    }

    while (j <= right)
    {
        indicesTmp[k] = indices[j];
        j++;
        k++;
    }

    for (i = left; i <= right; i++)
    {
        indices[i] = indicesTmp[i];
    }
}

template <>
__device__ void merge<char *>(char **keys, size_t *indices, size_t *indicesTmp,
                              int left, int mid, int right, bool ascending)
{
    int i = left;    
    int j = mid + 1; 
    int k = left;    

    while (i <= mid && j <= right)
    {
        if (operators<char *>(keys[indices[i]], keys[indices[j]], ascending))
        {
            indicesTmp[k] = indices[i];
            i++;
        }
        else
        {
            indicesTmp[k] = indices[j];
            j++;
        }
        k++;
    }

    while (i <= mid)
    {
        indicesTmp[k] = indices[i];
        i++;
        k++;
    }

    while (j <= right)
    {
        indicesTmp[k] = indices[j];
        j++;
        k++;
    }

    for (i = left; i <= right; i++)
    {
        indices[i] = indicesTmp[i];
    }
}

template <typename T>
__global__ void mergeSortKernel(T *keys, size_t *indices, size_t *indicesTmp,
                                int n, int width, bool ascending)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * 2 * width;

    if (left >= n)
        return;

    int mid = min(left + width - 1, n - 1);
    int right = min(left + 2 * width - 1, n - 1);

    merge<T>(keys, indices, indicesTmp, left, mid, right, ascending);
}

template __global__ void mergeSortKernel<float>(
    float *, size_t *, size_t *,
    int, int, bool);

template __global__ void mergeSortKernel<uint64_t>(
    uint64_t *, size_t *, size_t *,
    int, int, bool);

    
template __global__ void mergeSortKernel<char *>(
    char **, size_t *, size_t *,
    int, int, bool);
