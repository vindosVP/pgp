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
        return fabs(first) < fabs(second);
    }
};

//__global__ void swapLines(double* matrix, double* identityMatrix, int i, int j, int size) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int offset = gridDim.x * blockDim.x;
//
//    for (int k = idx; k < size; k += offset) {
//        double temp = matrix[k * size + i];
//        matrix[k * size + i] = matrix[k * size + j];
//        matrix[k * size + j] = temp;
//
//        temp = identityMatrix[k * size + i];
//        identityMatrix[k * size + i] = identityMatrix[k * size + j];
//        identityMatrix[k * size + j] = temp;
//    }
//}
//
//
//__global__ void nullifyDown(double* matrix, double* identityMatrix, int size, int x) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//    int offsetx = gridDim.x * blockDim.x;
//    int offsety = gridDim.y * blockDim.y;
//
//    for (int i = x + idx + 1; i < size; i += offsetx) {
//        for (int j = x + idy + 1; j < size; j += offsety) {
//            matrix[j * size + i] = - matrix[x * size + i] / matrix[x * size + x]
//                                   * matrix[j * size + x] + matrix[j * size + i];
//        }
//        for (int j = idy; j < size; j += offsety) {
//            identityMatrix[j * size + i] = - matrix[x * size + i] / matrix[x * size + x]
//                                           * identityMatrix[j * size + x] + identityMatrix[j * size + i];
//        }
//    }
//
//}
//
//__global__ void nullifyUp(double* matrix, double* identityMatrix, int size, int x) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    int idy = threadIdx.y + blockIdx.y * blockDim.y;
//    int offsetx = gridDim.x * blockDim.x;
//    int offsety = gridDim.y * blockDim.y;
//
//    for (int i = x - idx - 1; i >= 0; i -= offsetx) {
//        for (int j = idy; j < size; j += offsety) {
//            identityMatrix[j * size + i] = - matrix[x * size + i] / matrix[x * size + x]
//                                           * identityMatrix[j * size + x] + identityMatrix[j * size + i];
//        }
//    }
//}
//
//__global__ void divideIdentityMatrix(double* matrix, double* identityMatrix, int size) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//    int offsetx = gridDim.x * blockDim.x;
//    int offsety = gridDim.y * blockDim.y;
//
//    for (int i = idx; i < size; i += offsetx) {
//        for (int j = idy; j < size; j += offsety) {
//            identityMatrix[j * size + i] /= matrix[i * size + i];
//        }
//    }
//}

__device__ __host__ findMaxElement(double* matrix, thrust::device_ptr<double> pointer, int idx, int size) {
    return thrust::max_element(pointer + idx * size + idx, pointer + (idx + 1) * size, compare_value())
           - pointer - idx * size;
}

__global__ void kernel(double* matrix, double* identityMatrix, int size, thrust::device_ptr<double> pointer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    for (int col = 0; col < size - 1; col++) {
        const int max_idx = findMaxElement(matrix, pointer, col, size);
        if (max_idx != idx){
            // swapLines<<<256, 256>>>(dev_matrix, dev_identityMatrix, i, max_idx, size);
            // swapLines(double* matrix, double* identityMatrix, int i, int j, int size)

            // Свапаем местами строки (если максимальный элемент стоит не на главной диагонали)
            for (int k = idx; k < size; k += offsetx) {
                double tempMatrixValue = matrix[k * size + col];
                matrix[k * size + col] = matrix[k * size + max_idx];
                matrix[k * size + max_idx] = tempMatrixValue;

                double tempIdentityMatrixValue = identityMatrix[k * size + col];
                identityMatrix[k * size + col] = identityMatrix[k * size + max_idx];
                identityMatrix[k * size + max_idx] = tempIdentityMatrixValue;
            }
        }

        // Зануляем элементы ниже главного
        // makeDownNull<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size, i);
        // nullifyDown(double* matrix, double* identityMatrix, int size, int x)
        for (int i = col + idx + 1; i < size; i += offsetx) {
            for (int j = col + idy + 1; j < size; j += offsety) {
                matrix[j * size + i] = - matrix[col * size + i] / matrix[col * size + col]
                                       * matrix[j * size + col] + matrix[j * size + i];
            }
            for (int j = idy; j < size; j += offsety) {
                identityMatrix[j * size + i] = - matrix[col * size + i] / matrix[col * size + col]
                                               * identityMatrix[j * size + col] + identityMatrix[j * size + i];
            }
        }
    }

    for (int col = size - 1; col > 0; col--) {
        // Зануляем всё выше главной диагонали
        // makeUpNull<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size, i);
        // nullifyUp(double* matrix, double* identityMatrix, int size, int x)
        for (int i = col - idx - 1; col >= 0; col -= offsetx) {
            for (int j = idy; j < size; j += offsety) {
                identityMatrix[j * size + i] = - matrix[col * size + i] / matrix[col * size + col]
                                               * identityMatrix[j * size + col] + identityMatrix[j * size + i];
            }
        }
    }

    // Делим строки присоединённой матрицы на соотв. элемент с главной диагонали главной матрицы
    // divideIdentityMatrix<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size);
    for (int i = idx; i < size; i += offsetx) {
        for (int j = idy; j < size; j += offsety) {
            identityMatrix[j * size + i] /= matrix[i * size + i];
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
            cin >> matrix[j * size + i];
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            i != j
                ? identityMatrix[i * size + j] = 0.0
                : identityMatrix[i * size + j] = 1.0;
        }
    }

    double* dev_matrix = 0;
    double* dev_identityMatrix = 0;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * array_size));
    CSC(cudaMalloc(&dev_identityMatrix, sizeof(double) * array_size));
    CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * array_size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_identityMatrix, identityMatrix, sizeof(double) * array_size, cudaMemcpyHostToDevice));
    const thrust::device_ptr<double> pointer = thrust::device_pointer_cast(dev_matrix);

//    for (int i = 0; i < size - 1; i++) {
//        const int max_idx = thrust::max_element(pointer + i * size + i, pointer + (i + 1) * size, comparator) - pointer - i * size;
//        if (max_idx != i) swapLines<<<256, 256>>>(dev_matrix, dev_identityMatrix, i, max_idx, size);
//        makeDownNull<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size, i);
//    }
//
//    for (int i = size - 1; i > 0; i--) {
//        makeUpNull<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size, i);
//    }
//
//    divideIdentityMatrix<<<block(32, 16), thread(32, 16)>>>(dev_matrix, dev_identityMatrix, size);

    kernel<<<dim3(32, 16), dim3(32, 16)>>>(dev_matrix, dev_identityMatrix, size, pointer);

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

    cudaFree(dev_matrix);
    cudaFree(dev_identityMatrix);
    delete[] matrix;
    delete[] identityMatrix;

    return 0;
}
