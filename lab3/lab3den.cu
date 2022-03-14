#include <iostream>
#include <vector>
#include <string>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct matrix {
    double values[3][3];
};


struct v3 {
    double x, y, z;
};

struct p {
    int x, y;
};


v3 glob_avgs[32];
matrix glob_covs[32];

__constant__ v3 gpu_avgs[32];
__constant__ matrix gpu_covs[32];


__device__ double count_class_number(uchar4* pixel, int idx) {
    double sub[3];
    double matrixAns[3];
    double class_num = 0.0;

    for (int i = 0; i < 3; i++) {
        matrixAns[i] = 0;
    }

    sub[0] = pixel->x - gpu_avgs[idx].x;
    sub[1] = pixel->y - gpu_avgs[idx].y;
    sub[2] = pixel->z - gpu_avgs[idx].z;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrixAns[i] += gpu_covs[idx].values[j][i] * sub[j];
        }
        class_num -= sub[i] * matrixAns[i];
    }

    return class_num;
}

__global__ void mahalanobis_kernel(uchar4* image, int nc, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    for (int j = idy; j < height; j += offset_y) {
        for (int i = idx; i < width; i += offset_x) {
            int temp = 0;
            uchar4 p = image[i + j * width];
            double max_jc = count_class_number(&p, 0);

            for (int k = 1; k < nc; k++) {

                double next_jc = count_class_number(&p, k);

                if (next_jc > max_jc) {
                    temp = k;
                    max_jc = next_jc;
                }
            }

            image[i + j * width].w = (unsigned char)temp;
        }
    }
}

void matrixInverse(matrix &matr) {
    double minor1 =  (matr.values[1][1] * matr.values[2][2] - matr.values[2][1] * matr.values[1][2]);
    double minor2 = -(matr.values[1][0] * matr.values[2][2] - matr.values[2][0] * matr.values[1][2]);
    double minor3 =  (matr.values[1][0] * matr.values[2][1] - matr.values[2][0] * matr.values[1][1]);
    double minor4 = -(matr.values[0][1] * matr.values[2][2] - matr.values[2][1] * matr.values[0][2]);
    double minor5 =  (matr.values[0][0] * matr.values[2][2] - matr.values[2][0] * matr.values[0][2]);
    double minor6 = -(matr.values[0][0] * matr.values[2][1] - matr.values[2][0] * matr.values[0][1]);
    double minor7 =  (matr.values[0][1] * matr.values[1][2] - matr.values[1][1] * matr.values[0][2]);
    double minor8 = -(matr.values[0][0] * matr.values[1][2] - matr.values[1][0] * matr.values[0][2]);
    double minor9 = (matr.values[0][0] * matr.values[1][1] - matr.values[1][0] * matr.values[0][1]);

    double D = matr.values[0][0] * minor1 - matr.values[0][1] * (-minor2) + matr.values[0][2] * minor3;

    matr.values[0][0] = minor1 / D;
    matr.values[0][1] = minor4 / D;
    matr.values[0][2] = minor7 / D;
    matr.values[1][0] = minor2 / D;
    matr.values[1][1] = minor5 / D;
    matr.values[1][2] = minor8 / D;
    matr.values[2][0] = minor3 / D;
    matr.values[2][1] = minor6 / D;
    matr.values[2][2] = minor9 / D;
}

void pre_calculate(std::vector<std::vector<p>> &image, uchar4* pixels, int nc, int width) {
    std::vector<v3> a;
    std::vector<matrix> c;
    a.resize(32);
    c.resize(32);

    for (int i = 0; i < nc; i++) {
        int size = image[i].size();
        a[i].x = 0;
        a[i].y = 0;
        a[i].z = 0;
        for (int j = 0; j < size; j++) {
            p point = image[i][j];
            uchar4 pixel = pixels[point.x + point.y * width];
            a[i].x += pixel.x;
            a[i].y += pixel.y;
            a[i].z += pixel.z;
        }
        a[i].x /= size;
        a[i].y /= size;
        a[i].z /= size;
        for (int j = 0; j < size; j++) {
            p point = image[i][j];
            uchar4 pixel = pixels[point.y * width + point.x];
            c[i].values[0][0] += (pixel.x - a[i].x) * (pixel.x - a[i].x);
            c[i].values[0][1] += (pixel.x - a[i].x) * (pixel.y - a[i].y);
            c[i].values[0][2] += (pixel.x - a[i].x) * (pixel.z - a[i].z);
            c[i].values[1][0] += (pixel.y - a[i].y) * (pixel.x - a[i].x);
            c[i].values[1][1] += (pixel.y - a[i].y) * (pixel.y - a[i].y);
            c[i].values[1][2] += (pixel.y - a[i].y) * (pixel.z - a[i].z);
            c[i].values[2][0] += (pixel.z - a[i].z) * (pixel.x - a[i].x);
            c[i].values[2][1] += (pixel.z - a[i].z) * (pixel.y - a[i].y);
            c[i].values[2][2] += (pixel.z - a[i].z) * (pixel.z - a[i].z);
        }
        if (size > 1) {
            size = (double)(size - 1);
            for (auto& row : c[i].values) {
                for (auto& item : row) {
                    item /= size;
                }
            }
        }
        
        matrixInverse(c[i]);
        glob_avgs[i] = a[i];
        glob_covs[i] = c[i];
    }
}

int main() {
    std::string input_filename;
    std::string output_filename;
    int nc, class_count, width, height;
    uchar4* out_pixels;

    std::cin >> input_filename;
    std::cin >> output_filename;
    std::cin >> nc;

    std::vector<std::vector<p>> image;
    image.resize(nc);

    for (int i = 0; i < nc; i++) {
        std::cin >> class_count;
        image[i].resize(class_count);
        for (int j = 0; j < class_count; j++) {
            std::cin >> image[i][j].x >> image[i][j].y;
        }
    }

    FILE* input = fopen(input_filename.c_str(), "rb");
    FILE* output = fopen(output_filename.c_str(), "wb");
    fread(&width, sizeof(int), 1, input);
    fread(&height, sizeof(int), 1, input);
    uchar4* px = (uchar4*)malloc(sizeof(uchar4) * width * height);
    fread(px, sizeof(uchar4), width * height, input);
    fclose(input);

    pre_calculate(image, px, nc, width);
    CSC(cudaMemcpyToSymbol(gpu_avgs, glob_avgs, 32 * sizeof(v3)));
    CSC(cudaMemcpyToSymbol(gpu_covs, glob_covs, 32 * sizeof(matrix)));
    CSC(cudaMalloc(&out_pixels, sizeof(uchar4) * width * height));
    CSC(cudaMemcpy(out_pixels, px, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));
    mahalanobis_kernel<<<dim3(32, 32), dim3(32, 32)>>>(out_pixels, nc, width, height);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(px, out_pixels, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
    CSC(cudaFree(out_pixels));

    fwrite(&width, sizeof(int), 1, output);
    fwrite(&height, sizeof(int), 1, output);
    fwrite(px, sizeof(uchar4), width * height, output);
    fclose(output);

    free(px);
    return 0;
}
