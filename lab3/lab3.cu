#include <iostream>
#include <tuple>
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
    double matr[3][3];
};

__constant__ matrix dev_cov[16];

struct vec3 {
    double x;
    double y;
    double z;
};

__constant__ vec3 dev_avg[16];

struct point {
    int x;
    int y;
};


void inverseMatrix(matrix& matrix) {
    double M1 =  (matrix.matr[1][1] * matrix.matr[2][2] - matrix.matr[2][1] * matrix.matr[1][2]);
    double M2 = -(matrix.matr[1][0] * matrix.matr[2][2] - matrix.matr[2][0] * matrix.matr[1][2]);
    double M3 =  (matrix.matr[1][0] * matrix.matr[2][1] - matrix.matr[2][0] * matrix.matr[1][1]);
    double M4 = -(matrix.matr[0][1] * matrix.matr[2][2] - matrix.matr[2][1] * matrix.matr[0][2]);
    double M5 =  (matrix.matr[0][0] * matrix.matr[2][2] - matrix.matr[2][0] * matrix.matr[0][2]);
    double M6 = -(matrix.matr[0][0] * matrix.matr[2][1] - matrix.matr[2][0] * matrix.matr[0][1]);
    double M7 =  (matrix.matr[0][1] * matrix.matr[1][2] - matrix.matr[1][1] * matrix.matr[0][2]);
    double M8 = -(matrix.matr[0][0] * matrix.matr[1][2] - matrix.matr[1][0] * matrix.matr[0][2]);
    double M9 =  (matrix.matr[0][0] * matrix.matr[1][1] - matrix.matr[1][0] * matrix.matr[0][1]);
    
    std::vector<std::vector<double>> M = {{M1, M2, M3},
                                          {M4, M5, M6},
                                          {M7, M8, M9}};

    double D = matrix.matr[0][0] * M1 - matrix.matr[0][1] * (-M2) + matrix.matr[0][2] * M3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M[i][j] /= D;
            matrix.matr[i][j] = M[i][j];
        }
    }
}

__device__ __host__ void getRGB(uchar4* pixel, vec3& dest) {
    dest.x = pixel->x;
    dest.y = pixel->y;
    dest.z = pixel->z;
}

std::tuple<std::vector<vec3>, std::vector<matrix>> pre_calculate(const std::vector<std::vector<point>>& classes, uchar4* img, int nc, int width) {
    std::vector<vec3> avg;
    avg.resize(16);
    std::vector<matrix> cov;
    cov.resize(16);

    for (int i = 0; i < nc; i++) {
        avg[i].x = 0;
        avg[i].y = 0;
        avg[i].z = 0;
        for (int j = 0; j < classes[i].size(); j++) {
            vec3 RGB;
            uchar4 p = img[classes[i][j].x + classes[i][j].y * width];
            getRGB(&p, RGB);
            avg[i].x += RGB.x;
            avg[i].y += RGB.y;
            avg[i].z += RGB.z;
        }

        avg[i].x /= classes[i].size();
        avg[i].y /= classes[i].size();
        avg[i].z /= classes[i].size();

        for (int j = 0; j < classes[i].size(); j++) {
            vec3 RGB;
            uchar4 p = img[classes[i][j].x + classes[i][j].y * width];
            getRGB(&p, RGB);
            vec3 sub;
            sub.x = RGB.x - avg[i].x;
            sub.y = RGB.y - avg[i].y;
            sub.z = RGB.z - avg[i].z;

            cov[i].matr[0][0] = sub.x * sub.x;
            cov[i].matr[0][1] = sub.x * sub.y;
            cov[i].matr[0][2] = sub.x * sub.z;
            cov[i].matr[1][0] = sub.y * sub.x;
            cov[i].matr[1][1] = sub.y * sub.y;
            cov[i].matr[1][2] = sub.y * sub.z;
            cov[i].matr[2][0] = sub.z * sub.x;
            cov[i].matr[2][1] = sub.z * sub.y;
            cov[i].matr[2][2] = sub.z * sub.z;
        }
        if (classes[i].size() > 1) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    cov[i].matr[j][k] /= (double)(classes[i].size() - 1);
                }
            }
        }
    }

    for (auto& matrix : cov) {
        inverseMatrix(matrix);
    }

    return std::make_tuple(avg, cov);
}

__device__ double count_class_number(uchar4* p, int idx) {
    vec3 RGB;
    getRGB(p, RGB);
    double sub[3];
    sub[0] = RGB.x - dev_avg[idx].x;
    sub[1] = RGB.y - dev_avg[idx].y;
    sub[2] = RGB.z - dev_avg[idx].z;

    double ans = 0;
    double temp[3];
    for (int i = 0; i < 3; i++) {
	temp[i] = 0;
    }

    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            temp[col] += dev_cov[idx].matr[row][col] * sub[row];
        }
        ans += sub[col] * temp[col];
    }
    return (-1) * ans;
}

__global__ void mahalanobis_kernel(uchar4* img,
                                   int width,
                                   int height,
                                   int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int y = idx; y < height; y += offset_y) {
        for (int x = idy; x < width; x += offset_x) {
            uchar4 p = img[x + y * width];
            double max_jc = count_class_number(&p, 0);
            int idx = 0;
            for (int k = 1; k < nc; k++) {
                double next_jc = count_class_number(&p, k);
                if(next_jc > max_jc) {
                    idx = k;
                    max_jc = next_jc;
                }
            }
            img[x + y * width].w = (unsigned char)idx;
        }
    }
}

int main() {
    std::string in_filename;
    std::string out_filename;
    int nc;
    int pixel_count;
    int width;
    int height;

    std::cin >> in_filename;
    std::cin >> out_filename;
    std::cin >> nc;

    // Input data
    std::vector<std::vector<point>> class_points(nc);
    for (int i = 0; i < nc; ++i) {
        std::cin >> pixel_count;
        class_points[i].resize(pixel_count);
        for (int j = 0; j < pixel_count; ++j) {
            std::cin >> class_points[i][j].x >> class_points[i][j].y;
        }
    }

    // File open
    FILE* in  = fopen(in_filename.c_str(), "rb");
    FILE* out = fopen(out_filename.c_str(), "wb");
    fread(&width, sizeof(int), 1, in);
    fread(&height, sizeof(int), 1, in);
    uchar4* img = (uchar4*)malloc(sizeof(uchar4) * width * height);
    fread(img, sizeof(uchar4), width * height, in);
    fclose(in);

    // Pre calculating
    std::vector<vec3> avg;
    std::vector<matrix> cov;
    tie(avg, cov) = pre_calculate(class_points, img, nc, width);

    matrix covs[16];
    vec3 avgs[16];

    for (int i = 0; i < nc; i++) {
	covs[i] = cov[i];
	avgs[i] = avg[i];
    }


    CSC(cudaMemcpyToSymbol(dev_avg, avgs, 16 * sizeof(vec3)));
    CSC(cudaMemcpyToSymbol(dev_cov, covs, 16 * sizeof(matrix)));

    uchar4* out_img;
    CSC(cudaMalloc(&out_img, sizeof(uchar4) * width * height));
    CSC(cudaMemcpy(out_img, img, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));

    mahalanobis_kernel<<<dim3(16, 16), dim3(16, 16)>>>(out_img, width, height, nc);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(img, out_img, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    CSC(cudaFree(out_img));

    fwrite(&width, sizeof(int), 1, out);
    fwrite(&height, sizeof(int), 1, out);
    fwrite(img, sizeof(uchar4), width * height, out);
    fclose(out);

    free(img);

    return 0;
}
