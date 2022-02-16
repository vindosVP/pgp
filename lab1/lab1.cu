#include <iostream>
#include <cstdio>
#include <chrono>

using namespace std;

__global__ void kernel(double *vec, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
    for(int i = idx; i < n; i += offset) {
        vec[i] *= vec[i];
    }
}

int main() {
    size_t n;
    scanf("%d", &n);
    double *vec = (double *)malloc(sizeof(double) * n);
    for(size_t i = 0; i < n; i++)
        scanf("%lf", &vec[i]);

    double *dev_vec;
    cudaMalloc(&dev_vec, sizeof(double) * n);
    cudaMemcpy(dev_vec, vec, sizeof(double) * n, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<256, 256>>>(dev_vec, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);


    cudaMemcpy(vec, dev_vec, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(dev_vec);
    for(size_t i = 0; i < n; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
    free(vec);
    return 0;
}