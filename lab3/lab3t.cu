#include <iostream>
#include <vector>
#include <string>

using namespace std;

#define CUDA_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "ERROR: CUDA failed in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return(1); \
    } \
} \

typedef struct {
    int x;
    int y;
} point;

typedef struct {
    double x;
    double y;
    double z;
} vec3;

typedef struct {
    double data[3][3];
} matr;


__constant__ vec3 gpu_avgs[32];
__constant__ matr gpu_covs[32];

vec3 copy_a[32];
matr copy_c[32];

void inverseMatr(matr &cov_i) {
    double M1 =  (cov_i.data[1][1] * cov_i.data[2][2] - cov_i.data[2][1] * cov_i.data[1][2]);
    double M2 = -(cov_i.data[1][0] * cov_i.data[2][2] - cov_i.data[2][0] * cov_i.data[1][2]);
    double M3 =  (cov_i.data[1][0] * cov_i.data[2][1] - cov_i.data[2][0] * cov_i.data[1][1]);

    double M4 = -(cov_i.data[0][1] * cov_i.data[2][2] - cov_i.data[2][1] * cov_i.data[0][2]);
    double M5 =  (cov_i.data[0][0] * cov_i.data[2][2] - cov_i.data[2][0] * cov_i.data[0][2]);
    double M6 = -(cov_i.data[0][0] * cov_i.data[2][1] - cov_i.data[2][0] * cov_i.data[0][1]);

    double M7 =  (cov_i.data[0][1] * cov_i.data[1][2] - cov_i.data[1][1] * cov_i.data[0][2]);
    double M8 = -(cov_i.data[0][0] * cov_i.data[1][2] - cov_i.data[1][0] * cov_i.data[0][2]);
    double M9 =  (cov_i.data[0][0] * cov_i.data[1][1] - cov_i.data[1][0] * cov_i.data[0][1]);

    double minor[3][3] = {{M1, M4, M7},
                          {M2, M5, M8},
                          {M3, M6, M9}};

    double D = cov_i.data[0][0] * M1 - cov_i.data[0][1] * (-M2) + cov_i.data[0][2] * M3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            minor[i][j] /= D;
            cov_i.data[i][j] = minor[i][j];
        }
    }
}

__device__ __host__ void getColors(vec3 &colors, uchar4* pixel) {
    colors.x = pixel->x;
    colors.y = pixel->y;
    colors.z = pixel->z;
}

double cpuFindPixel(uchar4* pixel, int idx) {
    vec3 colors;
    getColors(colors, pixel);

    double diff[3];
    diff[0] = colors.x - copy_a[idx].x;
    diff[1] = colors.y - copy_a[idx].y;
    diff[2] = colors.z - copy_a[idx].z;

    double matrAns[3];
    matrAns[0] = 0;
    matrAns[1] = 0;
    matrAns[2] = 0;

    // 1x3 * 3x3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrAns[i] += copy_c[idx].data[j][i] * diff[j];
        }
    }

    double ans = 0.0;
    for (int i = 0; i < 3; ++i) {
        ans += diff[i] * matrAns[i];
    }

    return -ans;
}

__device__ double findPixel(uchar4* pixel, int idx) {
    vec3 colors;
    getColors(colors, pixel);

    double diff[3];
    diff[0] = colors.x - gpu_avgs[idx].x;
    diff[1] = colors.y - gpu_avgs[idx].y;
    diff[2] = colors.z - gpu_avgs[idx].z;

    double matrAns[3];
    matrAns[0] = 0;
    matrAns[1] = 0;
    matrAns[2] = 0;

    // 1x3 * 3x3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrAns[i] += gpu_covs[idx].data[j][i] * diff[j];
        }
    }

    double ans = 0.0;
    for (int i = 0; i < 3; ++i) {
        ans += diff[i] * matrAns[i];
    }
    return -ans;
}

void CPUmahalanobis_kernel(uchar4* pixels, int w, int h, int nc) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            uchar4 pixel = pixels[row * w + col];
            double mx = cpuFindPixel(&pixel, 0);
            int mIdx = 0;
            for (int i = 1; i < nc; ++i) {
                double tmp = cpuFindPixel(&pixel, i);
                if (mx < tmp) {
                    mx = tmp;
                    mIdx = i;
                }
            }
            pixels[row * w + col].w = (unsigned char)mIdx;
        }
    }
}

__global__ void mahalanobis_kernel(uchar4* pixels, int w, int h, int nc) {
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetY = gridDim.y * blockDim.y;
    int offsetX = gridDim.x * blockDim.x;

    for (int row = idY; row < h; row += offsetY) {
        for (int col = idX; col < w; col += offsetX) {
            uchar4 pixel = pixels[row * w + col];
            double mx = findPixel(&pixel, 0);
            int mIdx = 0;
            for (int i = 1; i < nc; ++i) {
                double tmp = findPixel(&pixel, i);
                if (mx < tmp) {
                    mx = tmp;
                    mIdx = i;
                }
            }
            pixels[row * w + col].w = (unsigned char)mIdx;
        }
    }
}

void calculate(vector<vector<point>> &v, uchar4* pixels, int nc, int w) {
    vector<vec3> avgs(32);
    vector<matr> covs(32);

    for (int i = 0; i < nc; ++i) {
        vec3 colors;
        avgs[i].x = 0;
        avgs[i].y = 0;
        avgs[i].z = 0;

        for (int j = 0; j < v[i].size(); ++j) {
            point point_ = v[i][j];
            uchar4 pixel = pixels[point_.y * w + point_.x];

            getColors(colors, &pixel);

            avgs[i].x += colors.x;
            avgs[i].y += colors.y;
            avgs[i].z += colors.z;
        }

        double val = v[i].size();
        avgs[i].x /= val;
        avgs[i].y /= val;
        avgs[i].z /= val;

        for (int j = 0; j < v[i].size(); ++j) {
            point point_ = v[i][j];
            uchar4 pixel = pixels[point_.y * w + point_.x];

            getColors(colors, &pixel);

            vec3 diff;
            diff.x = colors.x - avgs[i].x;
            diff.y = colors.y - avgs[i].y;
            diff.z = colors.z - avgs[i].z;

            matr tmp;

            // diff * diff.T
            tmp.data[0][0] = diff.x * diff.x;
            tmp.data[0][1] = diff.x * diff.y;
            tmp.data[0][2] = diff.x * diff.z;
            tmp.data[1][0] = diff.y * diff.x;
            tmp.data[1][1] = diff.y * diff.y;
            tmp.data[1][2] = diff.y * diff.z;
            tmp.data[2][0] = diff.z * diff.x;
            tmp.data[2][1] = diff.z * diff.y;
            tmp.data[2][2] = diff.z * diff.z;

            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    covs[i].data[k][l] += tmp.data[k][l];
                }
            }
        }

        if (v[i].size() > 1) {
            val = (double)(v[i].size() - 1);
            for (auto & k : covs[i].data) {
                for (double & l : k) {
                    l /= val;
                }
            }
        }
    }

    for (int i = 0; i < nc; ++i) {
        inverseMatr(covs[i]);
        copy_a[i] = avgs[i];
        copy_c[i] = covs[i];
    }
}

int main() {
    string nameIn;
    string nameOut;
    int nc, cs, w, h;

    cin >> nameIn;
    cin >> nameOut;
    cin >> nc;

    // Input data
    vector<vector<point>> v(nc);
    for (int i = 0; i < nc; ++i) {
        cin >> cs;
        v[i].resize(cs);
        for (int j = 0; j < cs; ++j) {
            cin >> v[i][j].x >> v[i][j].y;
        }
    }

    // File open
    FILE* in  = fopen(nameIn.c_str(), "rb");
    FILE* out = fopen(nameOut.c_str(), "wb");

    fread(&w, sizeof(int), 1, in);
    fread(&h, sizeof(int), 1, in);

    uchar4* pixels = (uchar4*)malloc(sizeof(uchar4) * w * h);

    fread(pixels, sizeof(uchar4), w * h, in);
    fclose(in);

    // Pre calculating
    calculate(v, pixels, nc, w);
    CUDA_ERROR(cudaMemcpyToSymbol(gpu_avgs, copy_a, 32 * sizeof(vec3)));
    CUDA_ERROR(cudaMemcpyToSymbol(gpu_covs, copy_c, 32 * sizeof(matr)));

    uchar4* out_pixels;
    CUDA_ERROR(cudaMalloc(&out_pixels, sizeof(uchar4) * w * h));
    CUDA_ERROR(cudaMemcpy(out_pixels, pixels, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

//    CPUmahalanobis_kernel(pixels, w, h, nc);

    mahalanobis_kernel<<<dim3(32, 32), dim3(32, 32)>>>(out_pixels, w, h, nc);
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaMemcpy(pixels, out_pixels, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaFree(out_pixels));

    fwrite(&w, sizeof(int), 1, out);
    fwrite(&h, sizeof(int), 1, out);
    fwrite(pixels, sizeof(uchar4), w * h, out);
    fclose(out);

    free(pixels);

    return 0;
}
