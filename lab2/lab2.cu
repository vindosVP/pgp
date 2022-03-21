#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

texture<uint32_t, 2, cudaReadModeElementType> input_photo;

struct pos {
	int32_t x;
	int32_t y;
};

__host__ void NewMap(uint32_t **dots, uint32_t height, uint32_t width) {
	*dots = new uint32_t[height * width];
}

__host__ void GetPhoto(uint32_t **input_dots, uint32_t *height, uint32_t *width, string input_file_name) {
	FILE *input_file = fopen(input_file_name.c_str(), "rb");

	uint32_t sizes[2];
	fread(sizes, sizeof(uint32_t), 2, input_file);
	*width = sizes[0];
	*height = sizes[1];
	
	uint32_t resolution = (*height) * (*width);

	NewMap(input_dots, *height, *width);
	fread(*input_dots, sizeof(uint32_t), resolution, input_file);
	fclose(input_file);
}

__host__ void WriteResult(uint32_t *output_dots, uint32_t height, uint32_t width, string output_file_name) {
	FILE *output_file = fopen(output_file_name.c_str(), "wb");

	uint32_t sizes[2] = {width, height};
	fwrite(sizes, sizeof(uint32_t), 2, output_file);
	
	uint32_t resolution = height * width;

	fwrite(output_dots, sizeof(uint32_t), resolution, output_file);
	fclose(output_file);
}

__device__ int32_t truepos(pos p, uint32_t height, uint32_t width) {
	return (p.x >= 0 && p.y >= 0 && p.x < (int32_t) width && p.y < (int32_t) height) ? (p.y	* (int32_t) width + p.x) : -1;
}

__device__ uint16_t ReturnMedian(uint16_t *color_array, uint16_t k) {
	uint16_t counter;
	uint8_t result;
	counter = 0;
	result = 0;

	for (uint16_t i = 0; i < 256; i++) {
		counter += color_array[i];
		if (counter > k / 2) {
			result = (uint8_t) i;
			break;
		}
	}

	return result;
}

__device__ uint32_t Sorted_arr(uint32_t height, uint32_t width, pos init, pos last) {
	
	uint16_t r_array[256], g_array[256], b_array[256], k;
	pos taken_dot;
    for (uint16_t i = 0; i < (uint16_t) 256; i++){
        g_array[i] = 0;
        b_array[i] = 0;
        r_array[i] = 0;
        k = 0;
    }
	for (taken_dot.x = init.x; taken_dot.x <= last.x; taken_dot.x++) {
		for (taken_dot.y = init.y; taken_dot.y <= last.y; taken_dot.y++) {
		    if (taken_dot.x >= 0 && taken_dot.y>= 0 && taken_dot.x < (int32_t) width && taken_dot.y < (int32_t) height){
		        continue;
		    }
			k++;
			r_array[(uint8_t) (tex2D(input_photo, taken_dot.x, taken_dot.y) >> (uint32_t) 8*0)  & 255]++;
			g_array[(uint8_t) (tex2D(input_photo, taken_dot.x, taken_dot.y) >> (uint32_t) 8*1)  & 255]++;
			b_array[(uint8_t) (tex2D(input_photo, taken_dot.x, taken_dot.y) >> (uint32_t) 8*2)  & 255]++;
		}
	}

	uint32_t medelements = (((uint32_t) ReturnMedian(r_array, k) << (uint32_t) 8*0) + 
	                        ((uint32_t) ReturnMedian(g_array, k) << (uint32_t) 8*1) + 
	                        ((uint32_t) ReturnMedian(b_array, k) << (uint32_t) 8*2) + 
	                        (uint32_t) 0 << (uint32_t) 8*3);
	
	return medelements;
}

__device__ void FindDots(pos p, uint32_t radius, uint32_t height, uint32_t width, uint32_t *output) {
	pos init;
	pos last;
	
	if (p.x + (int32_t) radius <= width - 1){
	    last.x = p.x + (int32_t) radius;
	}else{
	    last.x = width - 1;
	}
	
	if (p.y + (int32_t) radius <= height - 1){
	    last.y = p.y + (int32_t) radius;
	}else{
	    last.y = height - 1;
	}
	
	if (p.x - (int32_t) radius >= 0){
	    init.x = p.x - (int32_t) radius;
	}else{
	    init.x = 0;
	}
	
	if (p.y - (int32_t) radius >= 0){
	    init.y = p.y - (int32_t) radius;
	}else{
	    init.y = 0;
	}
	
	uint32_t array_sort = Sorted_arr(height, width, init, last);

	output[truepos(p, height, width)] = array_sort;
}


__global__ void kernel(uint32_t radius, uint32_t height, uint32_t width, uint32_t *output) {

	pos start; 
	pos move;
	pos p;

	start.x = blockIdx.x * blockDim.x + threadIdx.x;
	start.x = blockIdx.y * blockDim.y + threadIdx.y;
	move.x = gridDim.x * blockDim.x;
	move.y = gridDim.y * blockDim.y;

	for (p.x = start.x; p.x < width; p.x += move.x) {
		for (p.y = start.y; p.y < height; p.y += move.y) {
			if (p.x < width && p.y < height) {
				FindDots(p, radius, height, width, output);
			}
		}
	}
}


__host__ int main(void) {
	uint32_t radius;
	string input_file_name
	string output_file_name;
	uint32_t *output_dots;
	uint32_t *input_dots;
	uint32_t height, width;

	cin >> input_file_name >> output_file_name >> radius; 

	GetPhoto(&input_dots, &height, &width, input_file_name);
	NewMap(&output_dots, height, width);

	uint32_t *gpu_result_dots;

	cudaArray *gpu_input_dots;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uint32_t>();
	cudaMallocArray(&gpu_input_dots, &ch, width, height);
	cudaMemcpyToArray(gpu_input_dots, 0, 0, input_dots, sizeof(uint32_t) * height * width, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &gpu_result_dots, sizeof(uint32_t) * width * height);
	
	input_photo.addressMode[0] = cudaAddressModeClamp;
	input_photo.addressMode[1] = cudaAddressModeClamp;
	input_photo.channelDesc = ch;
	input_photo.filterMode = cudaFilterModePoint;
	input_photo.normalized = false;
	cudaBindTextureToArray(input_photo, gpu_input_dots, ch);

	kernel<<<512, 512>>>(radius, height, width, gpu_result_dots);

	cudaMemcpy(output_dots, gpu_result_dots, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(input_photo);
	cudaFree(gpu_result_dots);
	cudaFreeArray(gpu_input_dots);

	WriteResult(output_dots, height, width, output_file_name);

	return 0;
}
