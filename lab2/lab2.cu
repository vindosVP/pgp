#include <iostream>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

const uint32_t COUNTING_SORT_BASE = 256;
const uint32_t BLOCK_DIM = 32;

namespace Dots {
    const uint32_t R = 0;
    const uint32_t G = 8;
    const uint32_t B = 16;
    const uint32_t A = 24;
}

texture<uint32_t, 2, cudaReadModeElementType> Photo;


struct pos {
    int32_t x;
    int32_t y;
};

__device__ int32_t truepos(pos pos, uint32_t h, uint32_t w) {
    if ((pos.y >= 0 && pos.x >= 0 && pos.y < (int32_t) h && pos.x < (int32_t) w) == true){
        return pos.y * (int32_t) w + pos.x;
    }
    else{
        return -1;
    }
}

__device__ uint8_t findmedian(uint16_t *arr, uint16_t size) {
    uint16_t tmp = 0;
    uint8_t res = 0;
    for (uint16_t i = 0; i < COUNTING_SORT_BASE; i++) {
        tmp += arr[i];
        if (tmp > size / 2) {
            res = (uint8_t) i;
            break;
        }
    }

    return res;
}

__device__ uint32_t MakeDot(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t) r << Dots::R) + ((uint32_t) g << Dots::G) +
           ((uint32_t) b << Dots::B) + ((uint32_t) a << Dots::A);
}

__device__ uint8_t FindTrueDot(uint32_t dot, uint32_t x) {
    return (uint8_t) (dot >> x) & 255;
}

__device__ uint32_t FindMedDot(uint32_t h, uint32_t w,
                                   pos begin, pos end) {
    uint16_t r_arr[COUNTING_SORT_BASE];
    uint16_t g_arr[COUNTING_SORT_BASE];
    uint16_t b_arr[COUNTING_SORT_BASE];
    for (uint16_t i = 0; i < (uint16_t) COUNTING_SORT_BASE; i++) {
        r_arr[i] = 0;
        g_arr[i] = 0;
        b_arr[i] = 0;
    }

    pos now_pos;

    uint16_t f = 0;
    for (now_pos.x = begin.x; now_pos.x <= end.x; now_pos.x++) {
        for (now_pos.y = begin.y; now_pos.y <= end.y; now_pos.y++) {
            if (!(now_pos.y >= 0 && now_pos.x >= 0 && now_pos.y < (int32_t) h && now_pos.x < (int32_t) w)) {
                continue;
            }
            r_arr[
                    FindTrueDot(
                            tex2D(Photo, now_pos.x, now_pos.y), Dots::R)]++;
            g_arr[
                    FindTrueDot(
                            tex2D(Photo, now_pos.x, now_pos.y), Dots::G)]++;
            b_arr[
                    FindTrueDot(
                            tex2D(Photo, now_pos.x, now_pos.y), Dots::B)]++;

            f++;
        }
    }

    return MakeDot(findmedian(r_arr, f),
                     findmedian(g_arr, f),
                     findmedian(b_arr, f), 0);
}

__device__ void findlastdot(pos pos, uint32_t radius, uint32_t h, uint32_t w, uint32_t *map) {
    pos begin, end;
    begin.x = max(pos.x - (int32_t) radius, 0);
    begin.y = max(pos.y - (int32_t) radius, 0);
    end.x = min(pos.x + (int32_t) radius, w - 1);
    end.y = min(pos.y + (int32_t) radius, h - 1);

    map[truepos(pos, h, w)] = FindMedDot(h, w, begin, end);
}


__global__ void filter(uint32_t radius, uint32_t h, uint32_t w,
                             uint32_t *map) {

    pos begin, move;
    begin.x = blockIdx.x * blockDim.x + threadIdx.x;
    begin.y = blockIdx.y * blockDim.y + threadIdx.y;

    move.x = gridDim.x * blockDim.x;
    move.y = gridDim.y * blockDim.y;

    pos pos;
    for (pos.X = start.X; pos.x < w; pos.x += move.x) {
        for (pos.y = start.y; pos.y < h; pos.y += move.y) {
            if (pos.y < h && pos.x < w) {
                findlastdot(pos, radius, h, w, map);
            }
        }
    }
}


__host__ void NewMap(uint32_t **dot, uint32_t h, uint32_t w) {
    *dot = new uint32_t[h * w];
}
__host__ uint32_t getDot(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t) r << Dots::R) + ((uint32_t) g << Dots::G) +
           ((uint32_t) b << Dots::B) + ((uint32_t) a << Dots::A);
}

__host__ void Destroydots(uint32_t **dot) {
    delete [] (*dot);
    *dot = NULL;
}

__host__ void ReadPhoto(uint32_t **dot, uint32_t *h, uint32_t *w,
                                string f) {
    FILE *file = fopen(f.c_str(), "rb");
    uint32_t resolution[2];
    fread(resolution, sizeof(uint32_t), 2, f);
    *w = resolution[0];
    *h = resolution[1];

    uint32_t weight = (*h) * (*w);

    NewMap(dot, *h, *w);
    fread(*dot, sizeof(uint32_t), weight, f);
    fclose(f);
}

__host__ void WritePhoto(uint32_t *dot, uint32_t h, uint32_t w, string f) {
    FILE *file = fopen(f.c_str(), "wb");
    uint32_t resolution[2] = {w, h};
    fwrite(resolution, sizeof(uint32_t), 2, f);

    uint32_t weight = h * w;
    fwrite(dot, sizeof(uint32_t), weight, f);
    fclose(f);
}

__host__ void write(){
    dot[0] = getDot(1, 2, 3, 0);
    dot[1] = getDot(4, 5, 6, 0);
    dot[2] = getDot(7, 8, 9, 0);

    dot[3] = getDot(9, 8, 7, 0);
    dot[4] = getDot(6, 5, 4, 0);
    dot[5] = getDot(3, 2, 1, 0);

    dot[6] = getDot(0, 0, 0, 0);
    dot[7] = getDot(20, 20, 20, 0);
    dot[8] = getDotl(0, 0, 0, 0);
}

__host__ void MakePhoto() {
    uint32_t *dot;
    uint32_t h = 3;
    uint32_t w = 3;
    NewMap(&dot, *h, *w);

    string f = "in.data";
    write();

    FILE *file = fopen(f.c_str(), "wb");
    uint32_t resolution[2] = {w, h};
    fwrite(resolution, sizeof(uint32_t), 2, f);

    uint32_t weight = h * w;
    fwrite(dot, sizeof(uint32_t), weight, f);
    fclose(f);

    delete [] (*dot);
    *dot = NULL;
}

__host__ int main(void) {
    string file_in, file_out;
    uint32_t radius;

    cin >> file_in >> file_out >> radius;

    uint32_t *in_dot;
    uint32_t *out_dot;
    uint32_t h, w;

    ReadPhoto(&in_dot, &h, &w, file_in);

    NewMap(&dot, *h, *w);

    uint32_t *cuda_dot_out;

    cudaArray *cuda_dot_in;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uint32_t>();
    cudaMallocArray(&cuda_dot_in, &ch, w, h);
    cudaMemcpyToArray(cuda_dot_in, 0, 0, in_dot, sizeof(uint32_t) * h * w, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &cuda_dot_out, sizeof(uint32_t) * w * h);

    OriginalImage.addressMode[0] = cudaAddressModeClamp;
    OriginalImage.addressMode[1] = cudaAddressModeClamp;

    Photo.channelDesc = ch;
    Photo.filterMode = cudaFilterModePoint;
    Photo.normalized = false;
    cudaBindTextureToArray(Photo, cuda_dot_in, ch);


    dim3 threads_per_block(w, h);
    dim3 blocks_per_grid(1, 1);

    if (h * w > BLOCK_DIM * BLOCK_DIM){
        threads_per_block.x = BLOCK_DIM;
        threads_per_block.y = BLOCK_DIM;
        blocks_per_grid.x = ceil((double) (w) / (double)(threads_per_block.x));
        blocks_per_grid.y = ceil((double) (h) / (double)(threads_per_block.y));
    }

    filter<<<blocks_per_grid, threads_per_block>>>(radius, h, w,
                                                         cuda_dot_out);

//    cudaEvent_t syncEvent;
//
//    cudaEventCreate(&syncEvent);
//    cudaEventRecord(syncEvent, 0);
//    cudaEventSynchronize(syncEvent);

    cudaMemcpy(out_dot, cuda_dot_out, sizeof(uint32_t) * w * h, cudaMemcpyDeviceToHost);

//    cudaEventDestroy(syncEvent);

    cudaUnbindTexture(Photo);
    cudaFreeArray(cuda_dot_in);
    cudaFree(cuda_dot_out);
    WriteImageToFile(out_dot, h, w, file_out);

    return 0;
}