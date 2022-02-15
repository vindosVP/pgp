#include <iostream>
#include <iomanip>

using namespace std;

#define HANDLE_ERROR(err)                             \
    do {                                              \
        if (err != cudaSuccess) {                     \
            printf("ERROR: %s\n", cudaGetErrorString(err)); \
            exit(0);                                  \
        }                                             \
    } while (0)

__global__ void kernel(double *res, double *arr1, double *arr2, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока
    int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
    for(int i = idx; i < n; i += offset) {
        res[i] = arr1[i] * arr2[i];
    }
}

int main()
{
    std::ios_base::sync_with_stdio(false);

    int size = 0;
    std::cin >> size;

    double *vec1 = new double[size];
    double *vec2 = new double[size];
    double *res = new double[size];

    for (int i = 0; i < size; ++i) {
        std::cin >> vec1[i];
    }
    for (int i = 0; i < size; ++i) {
        std::cin >> vec2[i];
    }

    double *dev1, *dev2, *devRes;

    HANDLE_ERROR(cudaMalloc((void **) &dev1, sizeof(double) * size));
    HANDLE_ERROR(cudaMalloc((void **) &dev2, sizeof(double) * size));
    HANDLE_ERROR(cudaMalloc((void **) &devRes, sizeof(double) * size));

    HANDLE_ERROR(cudaMemcpy(dev1, vec1, sizeof(double) * size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev2, vec2, sizeof(double) * size, cudaMemcpyHostToDevice));

    kernel<<<256, 256>>>(dev1, dev2, devRes, size);
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpy(res, devRes, sizeof(double) * size, cudaMemcpyDeviceToHost));

    std::cout.precision(10);
    std::cout.setf(std::ios::scientific);
    for (int i = 0; i < size; ++i) {
        std::cout << res[i] << ' ';
    }
    std::cout << '\n';

    HANDLE_ERROR(cudaFree(dev1));
    HANDLE_ERROR(cudaFree(dev2));
    HANDLE_ERROR(cudaFree(devRes));

    delete[] vec1;
    delete[] vec2;
    delete[] res;
}
