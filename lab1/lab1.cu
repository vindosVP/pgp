#include <iostream>
#include <chrono>

using namespace std;

__global__ void kernel(double *res, double *arr1, double *arr2, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
    for(int i = idx; i < n; i += offset) {
        res[i] = arr1[i] * arr2[i];
    }
}

int main() {
    int size;
    scanf("%d", &size);
    double *res = (double *)malloc(sizeof(double) * size);
    double *vec1 = (double *)malloc(sizeof(double) * size);
    double *vec2 = (double *)malloc(sizeof(double) * size);
    for(int i = 0; i < size; i++) {
        scanf("%lf", &vec1[i]);
    }
    for(int i = 0; i < size; i++) {
        scanf("%lf", &vec2[i]);
    }

    double *devRes, *devVec1, *devVec2;
    cudaMalloc(&devRes, sizeof(double) * size);
    cudaMemcpy(devRes, res, sizeof(double) * size, cudaMemcpyHostToDevice);

    cudaMalloc(&devVec1, sizeof(double) * size);
    cudaMemcpy(devVec1, vec1, sizeof(double) * size, cudaMemcpyHostToDevice);

    cudaMalloc(&devVec2, sizeof(double) * size);
    cudaMemcpy(devVec2, vec2, sizeof(double) * size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernel<<<256, 256>>>(devRes, devVec1, devVec2, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    fprintf(stderr, "time = %f\n", time);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);


    // cudaMemcpy(res, devRes, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(devRes);
    cudaFree(devVec1);
    cudaFree(devVec2);
//    for(int i = 0; i < n; i++) {
//        printf("%f ", res[i]);
//    }
    printf("\n");
    free(res);
    free(vec1);
    free(vec2);
    return 0;
}
