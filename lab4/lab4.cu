#include <iostream>
#include <math.h>
#include <cmath>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
using namespace std;


class Compare{
public:
    __host__ __device__ bool operator()(const double a, const double b) const{
        return fabs(a) < fabs(b);
    }
};

__host__ __device__ void changematrix(double* matrix, int f, int n, int i, int j){
    int tmp;
    tmp = matrix[k * n + i];
    matrix[k * n + i] = matrix[k * n + j];
    matrix[k * n + j] = tmp;
}
 __host__ __device__ void changermatr(double* rmatr, int f, int n, int i, int j){
    int tmp;
     tmp = rmatr[k * n + i];
     rmatr[k * n + i] = rmatr[k * n + j];
     rmatr[k * n + j] = tmp;
}

__global__ void change(double* matrix, double* rmatr, int n, int i, int j) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    int f = 0;
    for (f = idx; f < n; f += offset) {
        changematrix(matrix, f, n, i, j);
        changermatr(rmatr, f, n, i, j);
    }
}

__global__ void findrdivision(double* matrix, double* rmatr, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int i, j;

    for (i = idx; i < n; i += offsetx) {
        for (j = idy; j < n; j += offsety) {
            rmatr[j * n + i] = rmatr[j * n + i] / matrix[i * n + i];
        }
    }
}


__global__ void setlesser(double* matrix, double* rmatr, int n, int x) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int i,j;
    double valuediv;

    for (i = x + 1 + idx; i < n; i += offsetx) {
        valuediv = (-1) * (matrix[x * n + i] / matrix[x * n + x]);
        for (j = x + 1 + idy; j < n; j += offsety) {
            matrix[j * n + i] = matrix[j * n + i] +
                    (valuediv * matrix[j * n + x]);
        }
        for (j = idy; j < n; j += offsety) {
            rmatr[j * n + i] = rmatr[j * n + i] +
                    (valuediv * rmatr[j * n + x]);
        }
    }
}

__global__ void setupper(double* matrix, double* rmatr, int n, int x) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int i, j;
    double valuediv;
    for (i = x - 1 - idx; i >= 0; i = i - offsetx) {
        valuediv = (-1) * (matrix[x * n + i] / matrix[x * n + x]);
        for (j = idy; j < n; j += offsety) {
            rmatr[j * n + i] = rmatr[j * n + i] +
                    (valuediv * rmatr[j * n + x]);
        }
    }
}

void readmatr(double* matrix, int n){
    int i;
    int j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cin >> matrix[j * n + i];
        }
    }
}

void rightmatr(double* rmatr, int n) {
    int i;
    int j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            if (i != j) {
                rmatr[i * n + j] = 0.0;
            } else {
                rmatr[i * n + j] = 1.0;
            }
        }
    }
}

void out(double* rmatr, int n) {
    int i;
    int j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cout << rmatr[j * n + i] << " ";
        }
        cout << "\n";
    }
}

int main() {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    cin >> n;
    int i;

    double *matrix = (double*)malloc(n * n * sizeof(double));
    double *rmatr = (double*)malloc(n * n * sizeof(double));

    readmatr(matrix, n);
    rightmatr(rmatr, n);

    double* cu_matrix = 0;
    double* cu_rmatr = 0;
    cudaMalloc(&cu_matrix, sizeof(double) * n * n);
    cudaMalloc(&cu_rmatr, sizeof(double) * n * n);
    cudaMemcpy(cu_matrix, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_rmatr, rmatr, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    const thrust::device_ptr<double> k = thrust::device_pointer_cast(cu_matrix);

    const Compare compare;
    for (i = 0; i < n - 1; ++i) {
        const int max = thrust::max_element(k + i * n + i, k + (i + 1) * n, compare) - k - i * n;
        if (max != i){
            change<<<256, 256>>>(cu_matrix, cu_rmatr, n, i, max);
        }
        setlesser<<<32, 32>>>(cu_matrix, cu_rmatr, n, i);
    }


    for (i = n - 1; i > 0; i--) {
        setupper<<<32, 32>>>(cu_matrix, cu_rmatr, n, i);
    }

    findrdivision<<<32, 32>>>(cu_matrix, cu_rmatr, n);


    cudaMemcpy(matrix, cu_matrix, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(rmatr, cu_rmatr, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    cudaFree(cu_matrix);
    cudaFree(cu_rmatr);

    cout << scientific;
    cout.precision(10);
    out(rmatr, n);

    delete[] matrix;
    delete[] rmatr;

    return 0;
}