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

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct compare_value {
    __device__ __host__ bool operator() (const double lhs, const double rhs) {
        return fabs(lhs) < fabs(rhs);
    }
};

__global__ void swap(double* matrix, double* identityMatrix, int col, int max_idx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += offset) {
        double tempMatrixValue = matrix[col + i * size];
        matrix[col + i * size] = matrix[max_idx + i * size];
        matrix[max_idx + i * size] = tempMatrixValue;

        double tempIdentityMatrixValue = identityMatrix[col + i * size];
        identityMatrix[col + i * size] = identityMatrix[max_idx + i * size];
        identityMatrix[max_idx + i * size] = tempIdentityMatrixValue;
    }
}

__global__ void nullifyDown(double* matrix, double* identityMatrix, int size, int col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int i = col + idx + 1; i < size; i += offsetx) {
        for (int j = col + idy + 1; j < size; j += offsety) {
            matrix[i + j * size] = - matrix[i + col * size] / matrix[col + col * size]
                                   * matrix[col + j * size] + matrix[i + j * size];
        }
        for (int j = idy; j < size; j += offsety) {
            identityMatrix[i + j * size] = - matrix[i + col * size] / matrix[col + col * size]
                                           * identityMatrix[col +  j * size] + identityMatrix[i + j * size];
        }
    }
}

__global__ void kernel(double* matrix, double* identityMatrix, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int col = size - 1; col > 0; col--) {
        // Зануляем всё выше главной диагонали
        for (int i = col - idx - 1; i >= 0; i -= offsetx) {
            for (int j = idy; j < size; j += offsety) {
                identityMatrix[i + j * size] = - matrix[i + col * size] / matrix[col + col * size]
                                               * identityMatrix[col + j * size] + identityMatrix[i + j * size];
            }
        }
    }

    // Делим строки присоединённой матрицы на соотв. элемент с главной диагонали главной матрицы
    for (int i = idx; i < size; i += offsetx) {
        for (int j = idy; j < size; j += offsety) {
            identityMatrix[i + j * size] /= matrix[i + i * size];
        }
    }
}


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    int size;
    cin >> size;
    const int array_size = size * size;

    double *matrix = (double*)malloc( array_size * sizeof(double));
    double *identityMatrix = (double*)malloc( array_size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cin >> matrix[i + j * size];
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            i != j
                ? identityMatrix[j + i * size] = 0.0
                : identityMatrix[j + i * size] = 1.0;
        }
    }

    double* dev_matrix = 0;
    double* dev_identityMatrix = 0;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * array_size));
    CSC(cudaMalloc(&dev_identityMatrix, sizeof(double) * array_size));
    CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * array_size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_identityMatrix, identityMatrix, sizeof(double) * array_size, cudaMemcpyHostToDevice));

    const thrust::device_ptr<double> pointer = thrust::device_pointer_cast(dev_matrix);

    for (int col = 0; col < size - 1; col++) {
        const int max_idx = thrust::max_element(pointer + col + col * size, pointer + (col + 1) * size, compare_value())
                            - col * size - pointer;
        if (max_idx != col) {
            // Свапаем местами строки (если максимальный элемент стоит не на главной диагонали)
            swap<<<32, 32>>>(dev_matrix, dev_identityMatrix, col, max_idx, size);
        }
        // Зануляем элементы ниже главного
        nullifyDown<<<dim3(32, 16), dim3(32, 16)>>>(dev_matrix, dev_identityMatrix, size, col);
    }

    // Зануляем всё выше главной диагонали
    // и делим строки присоединённой матрицы на соотв. элемент с главной диагонали левой матрицы
    kernel<<<dim3(32, 16), dim3(32, 16)>>>(dev_matrix, dev_identityMatrix, size);

    CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * array_size, cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(identityMatrix, dev_identityMatrix, sizeof(double) * array_size, cudaMemcpyDeviceToHost));

    cout << scientific;
    cout.precision(10);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << identityMatrix[i + j * size] << ' ';
        }
        cout << '\n';
    }

    CSC(cudaFree(dev_matrix));
    CSC(cudaFree(dev_identityMatrix));
    delete[] matrix;
    delete[] identityMatrix;

    return 0;
}
