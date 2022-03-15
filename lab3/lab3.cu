#include <iostream>
#include <vector>
#include <string>

using namespace std;

struct position{
    int x;
    int y;
} ;

struct vector{
    double x;
    double y;
    double z;
} ;

struct matrix{
    double dots[3][3];
} ;


__constant__ vector vec[32];
__constant__ matrix dots[32];

vector void_vec[32];
matrix void_matr[32];

void inverseMatr(matrix &matrix_vals) {
    double M1 =  (matrix_vals.dots[1][1] * matrix_vals.dots[2][2] - matrix_vals.dots[2][1] * matrix_vals.dots[1][2]);
    double M2 = -(matrix_vals.dots[1][0] * matrix_vals.dots[2][2] - matrix_vals.dots[2][0] * matrix_vals.dots[1][2]);
    double M3 =  (matrix_vals.dots[1][0] * matrix_vals.dots[2][1] - matrix_vals.dots[2][0] * matrix_vals.dots[1][1]);

    double M4 = -(matrix_vals.dots[0][1] * matrix_vals.dots[2][2] - matrix_vals.dots[2][1] * matrix_vals.dots[0][2]);
    double M5 =  (matrix_vals.dots[0][0] * matrix_vals.dots[2][2] - matrix_vals.dots[2][0] * matrix_vals.dots[0][2]);
    double M6 = -(matrix_vals.dots[0][0] * matrix_vals.dots[2][1] - matrix_vals.dots[2][0] * matrix_vals.dots[0][1]);

    double M7 =  (matrix_vals.dots[0][1] * matrix_vals.dots[1][2] - matrix_vals.dots[1][1] * matrix_vals.dots[0][2]);
    double M8 = -(matrix_vals.dots[0][0] * matrix_vals.dots[1][2] - matrix_vals.dots[1][0] * matrix_vals.dots[0][2]);
    double M9 =  (matrix_vals.dots[0][0] * matrix_vals.dots[1][1] - matrix_vals.dots[1][0] * matrix_vals.dots[0][1]);

    double minor[3][3] = {{M1, M4, M7},{M2, M5, M8},{M3, M6, M9}};

    double D = matrix_vals.dots[0][0] * M1 - matrix_vals.dots[0][1] * (-M2) + matrix_vals.dots[0][2] * M3;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            matrix_vals.dots[i][j] = minor[i][j] / D;
        }
    }
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

__device__ double findPixel(uchar4* dot, int id_X) {
    double div[3]
    double ans = 0.0;
    double Ans[3];
    Ans[0] = 0;
    Ans[1] = 0;
    Ans[2] = 0;

    div[0] = dot->x - vec[id_X].x;
    div[1] = dot->y - vec[id_X].y;
    div[2] = dot->z - vec[id_X].z;


    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Ans[i] += vec[id_X].dots[j][i] * div[j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        ans += div[i] * Ans[i];
    }
    return -ans;
}


__global__ void mahalanobis_kernel(uchar4* photo_dots, int width, int height, int nc) {
    int id_X = blockIdx.x * blockDim.x + threadIdx.x;
    int id_Y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_X = gridDim.x * blockDim.x;
    int offset_Y = gridDim.y * blockDim.y;

    for (int row = id_Y; row < h; row += offset_Y) {
        for (int col = id_X; col < w; col += offset_X) {
            uchar4 dot = photo_dots[row * w + col];
            double max = findPixel(&dot, 0);
            int timed = 0;
            for (int i = 1; i < nc; i++) {
                double next = findPixel(&dot, i);
                if (timed > mx) {
                    max = next;
                    timed = i;
                }
            }
            photo_dots[row * w + col].w = (unsigned char)timed;
        }
    }
}

void calculate(uchar4* photo_dots, vector<vector<point>> &input_photo, int nc, int width) {
    vector<vector> vec_vals.resize(32);
    vector<matrix> matrix_vals.resize(32);

    for (int i = 0; i < nc; i++) {
        vec3 colors;
        vec_vals[i].x = 0;
        vec_vals[i].y = 0;
        vec_vals[i].z = 0;

        for (int j = 0; j < input_photo[i].size(); j++) {
            position point = input_photo[i][j];
            uchar4 dots = photo_dots[width * point.y + point.x];

            vec_vals[i].x += dots.x;
            vec_vals[i].y += dots.y;
            vec_vals[i].z += dots.z;
        }

        vec_vals[i].x /= input_photo[i].size();
        vec_vals[i].y /= input_photo[i].size();
        vec_vals[i].z /= input_photo[i].size();

        for (int j = 0; j < input_photo[i].size(); ++j) {
            position point = input_photo[i][j];
            uchar4 dots = photo_dots[width * point.y * w + point.x];

            matrix timed;
            timed.dots[0][0] = (dots.x - vec_vals[i].x) * (dots.x - vec_vals[i].x);
            timed.dots[0][1] = (dots.x - vec_vals[i].x) * (dots.y - vec_vals[i].y);
            timed.dots[0][2] = (dots.x - vec_vals[i].x) * (dots.z - vec_vals[i].z);
            timed.dots[1][0] = (dots.y - vec_vals[i].y) * (dots.x - vec_vals[i].x);
            timed.dots[1][1] = (dots.y - vec_vals[i].y) * (dots.y - vec_vals[i].y);
            timed.dots[1][2] = (dots.y - vec_vals[i].y) * (dots.z - vec_vals[i].z);
            timed.dots[2][0] = (dots.z - vec_vals[i].z) * (dots.x - vec_vals[i].x);
            timed.dots[2][1] = (dots.z - vec_vals[i].z) * (dots.y - vec_vals[i].y);
            timed.dots[2][2] = (dots.z - vec_vals[i].z) * (dots.z - vec_vals[i].z);

            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    matrix_vals[i].dots[k][l] += timed.dots[k][l];
                }
            }
        }

        if (input_photo[i].size() > 1) {
            diff = (double)(input_photo[i].size() - 1);
            for (auto & k : matrix_vals[i].dots) {
                for (double & l : k) {
                    l /= diff;
                }
            }
        }
    }

    for (int i = 0; i < nc; ++i) {
        inverseMatr(matrix_vals[i]);
        void_vec[i] = vec_vals[i];
        void_matr[i] = matrix_vals[i];
    }
}

int main() {
    string output_file_name, input_file_name;
    int nc, width, height;

    cin >> input_file_name;
    cin >> output_file_name;
    cin >> nc;


    vector<vector<position>> input_photo;

    int class_pixels_number;
    for (int row = 0; row < nc; row++) {
        std::cin >> class_pixels;
        input_photo[row].resize(class_pixels);
        for (int column = 0; column < class_pixels_number; column++) {
            cin >> input_photo[row][column].x >> input_photo[row][column].y;
        }
    }

    FILE* input_file  = fopen(input_file_name.c_str(), "rb");
    FILE* output_file = fopen(output_file_name.c_str(), "wb");

    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, output_file);

    uchar4* photo_dots = (uchar4*)malloc(sizeof(uchar4) * width * height);

    fread(photo_dots, sizeof(uchar4), width * height, input_file);
    fclose(input_file);


    calculate(photo_dots, input_photo, nc, width);
    cudaMemcpyToSymbol(vec, void_vec, 32 * sizeof(vector));
    cudaMemcpyToSymbol(dots, void_matr, 32 * sizeof(matrix));

    uchar4* output_dots;
    cudaMalloc(&output_dots, sizeof(uchar4) * width * height);
    cudaMemcpy(output_dots, photo_dots, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice);



    mahalanobis_kernel<<<dim3(32, 32), dim3(32, 32)>>>(output_dots, width, height, nc);
    cudaGetLastError();
    cudaMemcpy(photo_dots, output_dots, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(output_dots);

    fwrite(&width, sizeof(int), 1, output_file);
    fwrite(&height, sizeof(int), 1, output_file);
    fwrite(photo_dots, sizeof(uchar4), width * height, output_file);
    fclose(output_file);

    free(photo_dots);

    return 0;
}