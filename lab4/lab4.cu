#include <stdio.h>
#include <math.h>
#include <float.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

struct compare_value {
    __device__ __host__ bool operator() (double lhs, double rhs) {
        return fabs(lhs) < fabs(rhs);
    }
};

__global__ void kernel(int division_row, int f, int n, double *input_matrix) {
    double div;
    int f1 = n - 1 - f;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idx + f; i < n + 1; i+= offsetx) {
        input_matrix[f + i * n] = input_matrix[f + i * n] * input_matrix[division_row + i * n] / (input_matrix[division_row + i * n] = input_matrix[f + i * n]);
    }
    
    for (int i = f + 1 + idx; i < n; i += offsetx) {
        div = input_matrix[i + f * n] / input_matrix[f + n * f];
        for (int j = f + 1 + idy; j < n + 1; j += offsety) {
            input_matrix[i + n * j] -= div * input_matrix[f + n * j];
        }
    }
    for (int i = idx; i < f1; i += offsetx) {
        div = input_matrix[i + f1 * n] / input_matrix[f1 + f1 * n];
        input_matrix[i + n * n] -= div * input_matrix[f1 + n * n];
    }   
}

__global__ void final_moves(int n, double *kernel_matrix) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += offsetx) {
        kernel_matrix[i + n * n] /= kernel_matrix[i + n *i];
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

    cudaMalloc(&gpu_matrix, sizeof(double) * (n + 1) * n);
    cudaMemcpy(gpu_matrix, input_matrix, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice);
    
    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> comparation = thrust::device_pointer_cast(gpu_matrix) + n * i;
        int division = thrust::max_element(comparation + i, comparation + n, compare_value()) - comparation;       
        kernel<<<32, 32>>>(division, i, n, gpu_matrix);
    }
    
    final_moves<<<32, 32>>>(n, gpu_matrix);
    
    cudaMemcpy(input_matrix + n * n, gpu_matrix + n * n, sizeof(double) * n, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        printf("%.10e ", input_matrix[n * n + i]);
    }
     
    cudaFree(gpu_matrix);
    free(input_matrix);
    return 0;
}
