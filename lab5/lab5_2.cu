#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <string>
                                                         
__global__ 
void atomic_histogram_add(int input_size, int* numbersGPU, int* histogramGPU) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
    for (int i = idx; i < input_size; i+=offset) {
		atomicAdd(histogramGPU + numbersGPU[i], 1);
	}
}

__global__ 
void GPU_sorting(int input_size, int* numbersGPU, int* output_numbers, int* histogramGPU) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
    int index;
	for (int i = idx; i < input_size; i+=offset) {
        index = atomicAdd(histogramGPU + numbersGPU[i], -1) - 1;
		output_numbers[index] = numbersGPU[i];
	}
}

int main() {
	int input_size, i;
    int scan_element = -1;
    int* numbersGPU;
    int* output_numbers;
	int* histogramGPU;
    std::string input_file_name, output_file_name;
    std::cin >> input_file_name;
    std::cin >> output_file_name;
    FILE* input_file  = fopen(input_file_name.c_str(), "rb");
    FILE* output_file = fopen(output_file_name.c_str(), "wb");
	fread(&input_size, sizeof(int), 1, input_file);
	int* input_numbers = (int*) malloc(sizeof(int) * input_size);
    fread(input_numbers, sizeof(int), input_size, input_file);
    for (i = 0; i < input_size; i++) {
        scan_element = (scan_element <= input_numbers[i]) ? input_numbers[i] : scan_element;
    }
	int histogram_idx = scan_element + 1;
    cudaMalloc(&histogramGPU, sizeof(int) * histogram_idx);
	cudaMemset(histogramGPU, 0, sizeof(int) * histogram_idx);
	cudaMalloc(&numbersGPU, sizeof(int) * input_size);
	cudaMemcpy(numbersGPU, input_numbers, sizeof(int) * input_size, cudaMemcpyHostToDevice);
	cudaMalloc(&output_numbers, sizeof(int) * input_size);

    atomic_histogram_add<<<64, 64>>>(input_size, numbersGPU, histogramGPU);
	cudaGetLastError();
	
	thrust::device_ptr<int> pointer = thrust::device_pointer_cast(histogramGPU);
    thrust::inclusive_scan(pointer, pointer + scan_element + 1, pointer);
	
	GPU_sorting<<<64, 64>>>(input_size, numbersGPU, output_numbers, histogramGPU);
	cudaGetLastError();
	
	cudaMemcpy(input_numbers, output_numbers, sizeof(int) * input_size, cudaMemcpyDeviceToHost);
	
	fwrite(input_numbers, sizeof(int), input_size, output_file);
    return 0;
}