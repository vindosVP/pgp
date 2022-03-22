#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

texture<uchar4, 2, cudaReadModeElementType> input_image;

__device__ int find_median_element(int *color_array, int count) {
    int tmp = 0;
    int medianelement;
    for (int i = 0; i < 256; i++) {
        tmp = tmp + color_array[i];
        if (tmp > count / 2) {
            medianelement = i;
            break;
        }            
    }
    return medianelement;
}

__global__ void kernel(int radius, int height, int width, uchar4* output_dots){
    uchar4* window = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4 p;

    int k, l, count;
    int r_dots[256], g_dots[256], b_dots[256];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y; 

    //!

    for (int x = idx; x < width; x += offsetx) {
        for (int y = idy; y < height; y += offsety) {
            count = 0;
            for (k = y - radius; k <= y + radius; k++) {
                for (l = x - radius; l <= x + radius; l++) {
                    if (l >= 0 && l < w && k >= 0 && k < h) {
                        p = tex2D(input_image, l, k);
                        r_dots[p.x] += 1;
                        g_dots[p.y] += 1;
                        b_dots[p.z] += 1;
                        count += 1;
                    }
                }
            }
            output_dots[x + y * width] = tex2D(input_image, x, y);
            output_dots[x + y * width].x = find_median_element(r_dots, count);
            output_dots[x + y * width].y = find_median_element(g_dots, count);
            output_dots[x + y * width].z = find_median_element(b_dots, count);
        }
    }
}


int main() {
    string input_filename, output_filename;
    int width, height, radius;
    std::cin >> input_filename;
    std::cin >> output_filename;
    std::cin >> radius;

    FILE * input_file = fopen(input_filename.c_str(), "rb");
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);
    uchar4* input_dots = (uchar4*)malloc(sizeof(uchar4) * width * height);
    fread(input_dots, sizeof(uchar4), width * height, input_file);

    cudaArray* image_data;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&image_data, &ch, width, height);

    cudaMemcpyToArray(image_data, 0, 0, input_dots, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice);
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;
    cudaBindTextureToArray(input_image, image_data, ch);

    uchar4* output_dots;
    cudaMalloc(&output_dots, sizeof(uchar4) * width * height);

    kernel << <dim3(16, 16), dim3(16, 32) >> > (radius, height, width, output_dots);
    cudaGetLastError();

    cudaMemcpy(input_dots, output_dots, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

    cudaUnbindTexture(input_image);

    cudaFreeArray(image_data);
    cudaFree(output_dots);

    FILE * output_file = fopen(output_filename.c_str(), "wb");
    
    fwrite(&height, sizeof(int), 1, output_file);
    fwrite(&width, sizeof(int), 1, output_file);

    fwrite(input_dots, sizeof(uchar4), width * height, output_file);
    
    fclose(output_file);
    fclose(input_file);

    free(input_dots);
    return 0;
}
