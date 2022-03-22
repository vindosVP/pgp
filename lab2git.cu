#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
using namespace std;

#define CSC(call)   \
do { \
cudaError_t res = call; \
if (res != cudaSuccess) { \
fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
__FILE__, __LINE__, cudaGetErrorString(res)); \
exit(0); \
} \
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;



 
__global__ void kernel(uchar4* out, int w, int h, int radius)
{

    int indexInX = blockDim.x * blockIdx.x + threadIdx.x;
    int indexInY = blockDim.y * blockIdx.y + threadIdx.y;
    int numCount = 0, iCol = 0, iRow = 0;
    uchar4 p;
    uchar4* window = (uchar4*)malloc(sizeof(uchar4) * w * h);
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int red[256];
    int green[256];
    int blue[256];
    int color[3];


    for (int x = indexInX; x < w; x += offsetx) {
        for (int y = indexInY; y < h; y += offsety) {

            numCount = 0;

            for (int i = 0; i < 256; i++) {
                red[i] = 0;
                green[i] = 0;
                blue[i] = 0;
            }

            for (iRow = y - radius; iRow <= y + radius; iRow++) {
                for (iCol = x - radius; iCol <= x + radius; iCol++) {
                    if (iCol >= 0 && iCol < w && iRow >= 0 && iRow < h) {
                        p = tex2D(tex, iCol, iRow);
                        red[p.x]++;
                        green[p.y]++;
                        blue[p.z]++;
                        numCount++;
                    }
                }
            }

            out[y * w + x] = tex2D(tex, x, y);

            int s = 0;
            for (int i = 0; i < 256; i++) {
                s += red[i];
                if (s > numCount / 2) {
                    color[0] = i;
                    break;
                }
               
            }
            s = 0;
            for (int i = 0; i < 256; i++) {
                s += green[i];
                if (s > numCount / 2) {
                    color[1] = i;
                    break;
                }
            }
            s = 0;
            for (int i = 0; i < 256; i++) {
                s += blue[i];
                if (s > numCount / 2) {
                    color[2] = i;
                    break;
                }
            }

            out[y * w + x].x = color[0];
            out[y * w + x].y = color[1];
            out[y * w + x].z = color[2];

        }
    }
}


int main() {
    string filename1, filename2;
    cin >> filename1 >> filename2;
    int w, h, radius = 3;
    cin >> radius;

    //FILE* f = fopen("input", "rb");
    FILE * f = fopen(filename1.c_str(), "rb");
    fread(&w, sizeof(int), 1, f);
    fread(&h, sizeof(int), 1, f);
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, f);
    fclose(f);

    // Подготовка данных для текстуры
    cudaArray* arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    // Подготовка текстурной ссылки, настройка интерфейса работы с данными


    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint; // Без интерполяции при обращении по дробным координатам
    tex.normalized = false; // Режим нормализации координат: без нормализации

    // Связываем интерфейс с данными
    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4* dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel << <dim3(16, 16), dim3(16, 32) >> > (dev_out, w, h, radius);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    f = fopen(filename2.c_str(), "wb");
    //f = fopen("out.data", "wb");
    fwrite(&w, sizeof(int), 1, f);
    fwrite(&h, sizeof(int), 1, f);
    fwrite(data, sizeof(uchar4), w * h, f);
    fclose(f);

    free(data);
    return 0;
}
