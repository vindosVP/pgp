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
    double m[3][3];
};

struct v3 {
    double x, y, z;
};

struct point {
    int x, y;
};

__constant__ v3 gpu_avgs[32];
__constant__ matrix gpu_covs[32];
v3 temp_avgs[32];
matrix temp_covs[32];

__device__ double count_class_number(uchar4* img, int index) {

    double matrixAns[3];
    double temp = 0.0;
    double diff[3];
    v3 colors;

    for (int i = 0; i < 3; i++) {
        matrixAns[i] = 0;
    }

    colors.x = img->x;
    colors.y = img->y;
    colors.z = img->z;
    diff[0] = colors.x - gpu_avgs[index].x;
    diff[1] = colors.y - gpu_avgs[index].y;
    diff[2] = colors.z - gpu_avgs[index].z;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrixAns[i] += gpu_covs[index].m[j][i] * diff[j];
        }
        temp -= diff[i] * matrixAns[i];
    }
    return temp;
}

void precalculate(std::vector<std::vector<point>> &pixels,
                  uchar4* img,
                  int nc,
                  int width) {
    std::vector<v3> avg;
    avg.resize(32);
    std::vector<matrix> cov;
    cov.resize(32);

    for (int i = 0; i < nc; ++i) {
        avg[i].x = 0;
        avg[i].y = 0;
        avg[i].z = 0;

        for (int j = 0; j < pixels[i].size(); j++) {
            v3 temp;
            uchar4 pixel = img[pixels[i][j].x + pixels[i][j].y * width];
            temp.x = pixel->x;
            temp.y = pixel->y;
            temp.z = pixel->z;
            avg[i].x += temp.x;
            avg[i].y += temp.y;
            avg[i].z += temp.z;
        }

        avg[i].x /= pixels[i].size();
        avg[i].y /= pixels[i].size();
        avg[i].z /= pixels[i].size();

        for (int j = 0; j < pixels[i].size(); ++j) {
            v3 temp;
            uchar4 pixel = img[pixels[i][j].x + pixels[i][j].y * width];
            temp.x = pixel->x;
            temp.y = pixel->y;
            temp.z = pixel->z;
            v3 sub;
            sub.x = temp.x - avg[i].x;
            sub.y = temp.y - avg[i].y;
            sub.z = temp.z - avg[i].z;

            matrix temp;

            temp.m[0][0] = sub.x * sub.x;
            temp.m[0][1] = sub.x * sub.y;
            temp.m[0][2] = sub.x * sub.z;
            temp.m[1][0] = sub.y * sub.x;
            temp.m[1][1] = sub.y * sub.y;
            temp.m[1][2] = sub.y * sub.z;
            temp.m[2][0] = sub.z * sub.x;
            temp.m[2][1] = sub.z * sub.y;
            temp.m[2][2] = sub.z * sub.z;

            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    cov[i].m[row][col] += temp.m[row][col];
                }
            }
        }
        if (pixels[i].size() > 1) {
            for (auto& matr : cov[i].m) {
                for (double& value : matr) {
                    value /= (double)(pixels[i].size() - 1);
                }
            }
        }

        inverseMatrix(cov[i]);
        temp_avgs[i] = avg[i];
        temp_covs[i] = cov[i];
    }
}

void inverseMatrix(matrix& matr) {
    double minor_1 =  (matr.m[1][1] * matr.m[2][2] - matr.m[2][1] * matr.m[1][2]);
    double minor_2 = -(matr.m[1][0] * matr.m[2][2] - matr.m[2][0] * matr.m[1][2]);
    double minor_3 =  (matr.m[1][0] * matr.m[2][1] - matr.m[2][0] * matr.m[1][1]);
    double minor_4 = -(matr.m[0][1] * matr.m[2][2] - matr.m[2][1] * matr.m[0][2]);
    double minor_5 =  (matr.m[0][0] * matr.m[2][2] - matr.m[2][0] * matr.m[0][2]);
    double minor_6 = -(matr.m[0][0] * matr.m[2][1] - matr.m[2][0] * matr.m[0][1]);
    double minor_7 =  (matr.m[0][1] * matr.m[1][2] - matr.m[1][1] * matr.m[0][2]);
    double minor_8 = -(matr.m[0][0] * matr.m[1][2] - matr.m[1][0] * matr.m[0][2]);
    double minor_9 =  (matr.m[0][0] * matr.m[1][1] - matr.m[1][0] * matr.m[0][1]);

    double minor[3][3] = {{minor_1, minor_4, minor_7},
                          {minor_2, minor_5, minor_8},
                          {minor_3, minor_6, minor_9}};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matr.m[i][j] = minor[i][j] / matr.m[0][0] * minor_1
                           - matr.m[0][1] * (-minor_2)
                           + matr.m[0][2] * minor_3;
        }
    }
}

__global__ void mahalanobis_distance(uchar4* img, int width, int height, int nc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = gridDim.x * blockDim.x;
    int offset_y = gridDim.y * blockDim.y;

    for (int y = idy; y < height; y += offset_y) {
        for (int x = idx; x < width; x += offset_x) {
            
            uchar4 p = img[x + y * width];
            double max_jc = count_class_number(&p, 0);
            
            int temp = 0;
            for (int i = 1; i < nc; ++i) {
                double next_jc = count_class_number(&p, i);
                if (max_jc < next_jc) {
                    temp = i;
                    max_jc = next_jc;
                }
            }
            
            img[x + y * width].w = (unsigned char)temp;
        }
    }
}


int main() {
    string nameIn;
    string nameOut;
    int nc;
    int pixel_count;
    int width
    int height;
    std::cin >> nameIn;
    std::cin >> nameOut;
    std::cin >> nc;

    std::vector<std::vector<point>> pixels(nc);
    for (int i = 0; i < nc; ++i) {
        std::cin >> pixel_count;
        pixels[i].resize(pixel_count);
        for (int j = 0; j < pixel_count; ++j) {
            std::cin >> pixels[i][j].x >> pixels[i][j].y;
        }
    }

    FILE* in_file = fopen(nameIn.c_str(), "rb");
    FILE* out_file = fopen(nameOut.c_str(), "wb");
    fread(&w, sizeof(int), 1, in_file);
    fread(&h, sizeof(int), 1, in_file);

    uchar4* img = (uchar4*)malloc(sizeof(uchar4) * width * height);
    fread(img, sizeof(uchar4), width * height, in_file);
    fclose(in_file);

    precalculate(pixels, img, nc, width);
    CSC(cudaMemcpyToSymbol(gpu_avgs, temp_avgs, 32 * sizeof(v3)));
    CSC(cudaMemcpyToSymbol(gpu_covs, temp_covs, 32 * sizeof(matrix)));
    uchar4* out_img;
    CSC(cudaMalloc(&out_img, sizeof(uchar4) * width * height));
    CSC(cudaMemcpy(out_img, img, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));
    mahalanobis_kernel<<<dim3(32, 32), dim3(32, 32)>>>(out_img, width, height, nc);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(img, out_img, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    fwrite(&width, sizeof(int), 1, out_file);
    fwrite(&width, sizeof(int), 1, out_file);
    fwrite(img, sizeof(uchar4), width * height, out_file);
    fclose(out_file);

    CSC(cudaFree(out_img));
    free(img);

    return 0;
}
