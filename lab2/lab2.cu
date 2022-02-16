#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

namespace Pixel {
    const uint32_t R = 8 * 0;
    const uint32_t G = 8 * 1;
    const uint32_t B = 8 * 2;
    const uint32_t A = 8 * 3;
}

texture<uint32_t, 2, cudaReadModeElementType> Photo;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__device__ uint8_t Pixel(uint32_t pixel, uint32_t element) {
    return (uint8_t) (pixel >> element) & 255;
}

__device__ uint8_t PixelFromCountArray(uint16_t *arr, uint16_t size) {
    uint16_t count = 0;
    uint8_t result = 0;
    for (uint16_t i = 0; i < 256; i++) {
        count += arr[i];
        if (count > size / 2) {
            result = (uint8_t) i;
            break;
        }
    }

    return result;
}

__device__ uint8_t PixelMaker(uint32_t pixel, uint32_t element) {
    return (uint8_t) (pixel >> element) & 255;
}

__device__ uint32_t R_DOT(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t) red << Pixel::R) + ((uint32_t) green << Pixel::G) +
           ((uint32_t) blue << Pixel::B) + ((uint32_t) alpha << Pixel::A);
}

__device__ uint32_t FindMed(uint32_t height, uint32_t width, uint32_t start_x,uint32_t start_y, uint32_t end_x, uint32_t end_y) {
    uint16_t red_arr[256];
    uint16_t green_arr[256];
    uint16_t blue_arr[256];
    for (uint16_t i = 0; i < (uint16_t) 256; i++) {
        red_arr[i] = 0;
        green_arr[i] = 0;
        blue_arr[i] = 0;
    }

    uint32_t pos_now_x;
    uint32_t pos_now_y;

    uint16_t size = 0;

    for (pos_now_x = start_x; pos_now_x <= end_x; pos_now_x++) {
        for (pos_now_y = start_y; pos_now_y <= end_y; pos_now_y++) {
            if (!CheckPos(pos_now_x, pos_now_y, h, w)) {
                continue;
            }
            red_arr[
                    PixelMaker(
                            tex2D(Photo, pos_now_x, pos_now_y), Pixel::R)]++;
            green_arr[
                    PixelMaker(
                            tex2D(Photo, pos_now_x, pos_now_y)), Pixel::G)]++;
            blue_arr[
                    PixelMaker(
                            tex2D(Photo, pos_now_x, pos_now_y)), Pixel::B)]++;

            size++;
        }
    }
    return R_DOT(GetMedianElementFromCountArray(red_arr, size),
                     GetMedianElementFromCountArray(green_arr, size),
                     GetMedianElementFromCountArray(blue_arr, size), 0);
}

__device__ int32_t CheckPos(uint32_t x, uint32_t y, uint32_t h, uint32_t w) {
    return ((x >= 0 && y >= 0 && x < (int32_t) w && y < (int32_t) h)) ? (y * (int32_t) w + x) : -1;
}

__device__ void TakePoint(uint32_t x,uint32_t y, uint32_t r, uint32_t h, uint32_t w, uint32_t *dev_output) {
    uint32_t start_x, start_y, end_x, end_y;
    start_x = max(x - (int32_t) r, 0);
    start_y = max(y - (int32_t) r, 0);
    end_x = min(x + (int32_t) r, w - 1);
    end_y = min(y + (int32_t) r, h - 1);

    dev_output[CheckPos(x, y, h, w)] = FindMed(h, w, start_x, start_y, end_x, end_y);
}

__global__ void median(uint32_t r, uint32_t h, uint32_t w, uint32_t *dev_output) {

    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;
    int x_offset = gridDim.x * blockDim.x;
    int y_offset = gridDim.y * blockDim.y;
    int x,y;
    for (x = xid; x < w; x += x_offset) {
        for (y = yid; y < h;  += y_offset) {
            if (x < w && y < h) {
                //cuPrintf("\n%d:%d\n", pos.X, pos.Y);
                TakePoint(x, y, r, h, w, dev_output);
            }
        }
    }
}

__host__ void MakePixels(uint32_t **pixel, uint32_t h, uint32_t w) {
    *pixel = new uint32_t[h * w];
}

__host__ void DeletePixels(uint32_t **pixel) {
    delete [] (*pixel);
    *pixel = NULL;
}

__host__ void PhotoFile(uint32_t **pixel, uint32_t *h, uint32_t *w, string name) {
    FILE *f = fopen(name.c_str(), "rb");
    uint32_t s[2] = {w, h};
    fread(s, sizeof(uint32_t), 2, f);

    uint32_t size = (*h) * (*);

    MakePixels(pixel, *h, *w);
    fread(*pixel, sizeof(uint32_t), size, f);
    fclose(f);
}

__host__ void OutFile(uint32_t *pixel, uint32_t h, uint32_t w, string name) {
    FILE *f = fopen(name.c_str(), "wb");
    uint32_t s[2] = {w, h};
    fwrite(s, sizeof(uint32_t), 2, f);

    uint32_t size = h * w;
    fwrite(pixel, sizeof(uint32_t), size, f);
    fclose(f);
}

__host__ void Makefile(string name) {
    uint32_t *pixel;
    uint32_t height = 3;
    uint32_t width = 3;
    MakePixels(&pixel, height, width);

    pixel[0] = SetPixel(1, 2, 3, 0);
    pixel[1] = SetPixel(4, 5, 6, 0);
    pixel[2] = SetPixel(7, 8, 9, 0);

    pixel[3] = SetPixel(9, 8, 7, 0);
    pixel[4] = SetPixel(6, 5, 4, 0);
    pixel[5] = SetPixel(3, 2, 1, 0);

    pixel[6] = SetPixel(0, 0, 0, 0);
    pixel[7] = SetPixel(20, 20, 20, 0);
    pixel[8] = SetPixel(0, 0, 0, 0);

    WriteImageToFile(pixel, h, w, name);
    DestroyPixelMap(&pixel);
}

__host__ int main() {
    string input_file, output_file;
    uint32_t w, h;
    uint32_t r;
    cin >> input_file >> output_file >> r;
    uint32_t *in_pixel
    uint32_t *out_pixel

    PhotoFile(&in_pixel, &h, &w, input_file);
    MakePixels(&out_pixel, h, w);

    uint32_t  *dev_out;

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpyToArray(arr, 0, 0, in_pixel, sizeof(uint32_t) * w * h, cudaMemcpyHostToDevice));
    cudaMalloc((void**) &dev_out, sizeof(uint32_t) * w * h);
    // Подготовка текстурной ссылки, настройка интерфейса работы с данными
    tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
    tex.normalized = false;						// Режим нормализации координат: без нормализации

    // Связываем интерфейс с данными
    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    median<<<dim3(32, 32), dim3(16, 16)>>>(r, h, w, dev_out);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    OutFile(out_pixel, h, w, out_pixel)

    DestroyPixelMap(&pixel_in);
    DestroyPixelMap(&pixel_out);

    DeletePixels(&in_pixel)
    DeletePixels(&out_pixel)

    return 0;
}