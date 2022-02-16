#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ float brightness(int4 point) {
    return 0.299 * float(point.x) + 0.587 * float(point.y) + 0.114 * float(point.z);
}

__device__ int4 summ(uchar4 p1, uchar4 p2, uchar4 p3) {
    int4 result;
    result.x = int(p1.x) + int(p2.x) + int(p3.x);
    result.y = int(p1.y) + int(p2.y) + int(p3.y);
    result.z = int(p1.z) + int(p2.z) + int(p3.z);
    result.w = 0;
    return result;
}

__device__ int4 sub(int4 first, int4 second) {
    int4 result;
    result.x = first.x - second.x;
    result.y = first.y - second.y;
    result.z = first.z - second.z;
    result.w = 0;
    return result;
}

__device__ float op_prewit(uchar4 z[9]) {
    int4 G_x = sub(summ(z[6],  z[7],  z[8]), summ(z[0],  z[1], z[2]));
    int4 G_y = sub(summ(z[2],  z[5],  z[8]), summ(z[0], z[3], z[6]));
    return sqrt(float(brightness(G_x) * brightness(G_x)) + float(brightness(G_y) * brightness(G_y)));
}

__global__ void kernel(uchar4* out, uint32_t w, uint32_t h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    for (int x = idx; x < w; x += offset_x) {
        for (int y = idy; y < h; y += offset_y) {
            uchar4 z[9];
            z[0] = tex2D(tex, x - 1, y - 1);
            z[1] = tex2D(tex, x, y - 1);
            z[2] = tex2D(tex, x + 1, y - 1);
            z[3] = tex2D(tex, x - 1, y);
            z[4] = tex2D(tex, x, y);
            z[5] = tex2D(tex, x + 1, y);
            z[6] = tex2D(tex, x - 1, y + 1);
            z[7] = tex2D(tex, x, y + 1);
            z[8] = tex2D(tex, x + 1, y + 1);
            float result = op_prewit(z);
            if (result > 255) {
                result = 255;
            }
            unsigned char byte = result;
            out[x + y * w] = make_uchar4(byte, byte, byte, z[4].w);
        }
    }
}

int main() {
    int w, h;
    char input_filename[256];
    char output_filename[256];
    scanf("%s", input_filename);
    scanf("%s", output_filename);
    FILE *fp = fopen(input_filename, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    // Подготовка данных для текстуры
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

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

    kernel<<<dim3(32, 32), dim3(16, 16)>>>(dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output_filename, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}
