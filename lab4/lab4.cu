#include <stdio.h>
#include <math.h>
#include <float.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>



#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

struct compare_value {
    __device__ __host__ bool operator() (double lhs, double rhs) {
        return fabs(lhs) < fabs(rhs);
    }
};

__global__ void subtract_row(double *matrix, int n, int column) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int i, j;
    
    double coeff;
    double divisor = matrix[column * n + column];
    for (i = 1 + column + idx; i < n; i += offsetx) {
        coeff = matrix[column * n + i] / divisor;
        for (j = 1 + column + idy; j < n + 1; j += offsety) {
            matrix[j * n + i] -= coeff * matrix[j * n + column];
        }
    }
}

__global__ void reverse_subtract_row(double *matrix, int n, int column) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    
    double coeff;
    double divisor = matrix[column * n + column];
    for (i = idx; i < column; i += offsetx) {
        coeff = matrix[column * n + i] / divisor;
        matrix[n * n + i] -= coeff * matrix[n * n + column];
    }
}

__global__ void normalize(double *matrix, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    
    for (i = idx; i < n; i += offsetx) {
        matrix[n * n + i] /= matrix[i * n + i];
    }
}

__global__ void swap_rows(int max_row, int n, int column, double *matrix) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    int i;
    double tmp;
    
    for (i = idx + column; i < n + 1; i+= offsetx) {
        tmp = matrix[i * n + column];
        matrix[i * n + column] = matrix[i * n + max_row];
        matrix[i * n + max_row] = tmp;
    }
}

int main() {
    int n;
    std::cin >> n;

    double *input_matrix = (double *) malloc(sizeof(double *) * (n + 1) * n);
    double *gpu_matrix;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> input_matrix[i + j * n];
            }
        }

    for (int i = 0; i < n; i++) {
        std::cin >> input_matrix[i + n * n];
    }

    CSC(cudaMalloc(&gpu_matrix, sizeof(double) * (n + 1) * n));
    CSC(cudaMemcpy(gpu_matrix, input_matrix, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice));
    
    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> comparation = thrust::device_pointer_cast(gpu_matrix) + n * i;
        int division = thrust::max_element(comparation + i, comparationx + n, compare_value()) - comparation;
        
        swap_rows<<<32, 32>>>(division, n, i, gpu_matrix);
        subtract_row<<<dim3(32, 32), dim3(32, 32)>>>(gpu_matrix, n, i);
    }
    
    for (int i = 0; i < n; ++i) { 
        reverse_subtract_row<<<32, 32>>>(gpu_matrix, n, n - 1 - i);
    }
    
    normalize<<<32, 32>>>(gpu_matrix, n);
    
    CSC(cudaMemcpy(input_matrix + n * n, gpu_matrix + n * n, sizeof(double) * n, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; ++i) {
        printf("%.10e ", input_matrix[n * n + i]);
    }
     
    CSC(cudaFree(gpu_matrix));
    free(input_matrix);
    return 0;
}
