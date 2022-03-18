#include <iostream>
#include <vector>
#include <string>

#define CSC(call) \
do { \
	cudaError_t res = call;  \
	if (res != cudaSuccess){ \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(res));  \
		exit(0);  \
	} \
} while(0)

using namespace std;

struct matrix{
    double dots[3][3];
};

struct position{
    int x,y;
};

struct vec_st{
    double x, y, z;
};

__constant__ vec_st vec[32];
__constant__ matrix dots[32];

vec_st void_vec[32];
matrix void_matr[32];

void start(uchar4* photo_dots, vector<vector<position>> &input_photo, int nc, int width) {
    vector<vec_st> vec_values;
    vec_values.resize(32);
    vector<matrix> matrix_values;
    matrix_values.resize(32);

    for (int i = 0; i < nc; i++) {

        vec_values[i].x = 0;
        vec_values[i].y = 0;
        vec_values[i].z = 0;

        for (int j = 0; j < input_photo[i].size(); j++) {
            position point = input_photo[i][j];
            uchar4 dots = photo_dots[width * point.y + point.x];

            vec_values[i].x = vec_values[i].x + dots.x;
            vec_values[i].y = vec_values[i].Y + dots.y;
            vec_values[i].z = vec_values[i].z + dots.z;
        }

        vec_values[i].x = vec_values[i].x / input_photo[i].size();
        vec_values[i].y = vec_values[i].y / input_photo[i].size();
        vec_values[i].z = vec_values[i].z / input_photo[i].size();

        for (int f = 0; f < input_photo[i].size(); f++) {
            position point = input_photo[i][f];
            uchar4 dots = photo_dots[width * point.y + point.x];

            matrix timed;
            timed.dots[0][0] = (dots.x - vec_values[i].x) * (dots.x - vec_values[i].x);
            timed.dots[0][1] = (dots.x - vec_values[i].x) * (dots.y - vec_values[i].y);
            timed.dots[0][2] = (dots.x - vec_values[i].x) * (dots.z - vec_values[i].z);
            timed.dots[1][0] = (dots.y - vec_values[i].y) * (dots.x - vec_values[i].x);
            timed.dots[1][1] = (dots.y - vec_values[i].y) * (dots.y - vec_values[i].y);
            timed.dots[1][2] = (dots.y - vec_values[i].y) * (dots.z - vec_values[i].z);
            timed.dots[2][0] = (dots.z - vec_values[i].z) * (dots.x - vec_values[i].x);
            timed.dots[2][1] = (dots.z - vec_values[i].z) * (dots.y - vec_values[i].y);
            timed.dots[2][2] = (dots.z - vec_values[i].z) * (dots.z - vec_values[i].z);

            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    matrix_values[i].dots[row][column] += timed.dots[row][column];
                }
            }
        }

        double diff;
        if (input_photo[i].size() > 1) {
            diff = (double)(input_photo[i].size() - 1);
            for (auto & k : matrix_values[i].dots) {
                for (double & l : k) {
                    l /= diff;
                }
            }
        }

        reversation(matrix_values[i]);
        void_vec[i] = vec_values[i];
        void_matr[i] = matrix_values[i];
    }
}

__device__ double get_current_pos(uchar4* dot, int f) {
    double arr_points[3], arr_result[3];

    for (i = 0, i < 3, i++){
        arr_result[i] = 0;
    }

    arr_points[0] = dot->x;
    arr_points[1] = dot->y ;
    arr_points[2] = dot->z ;

    arr_points[0] -= vec[f].x;
    arr_points[1] -= vec[f].y;
    arr_points[2] -= vec[f].z;

    for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
            arr_result[row] = arr_result[row] + arr_points[column] * dots[f].dots[column][row];
        }
    }

    double tmp = 0.0;

    for (int i = 0; i < 3; i++) {
        tmp += arr_points[i] * arr_result[i];
    }
    tmp = tmp * (-1);
    return tmp;
}


__global__ void kernel(uchar4* photo_dots, int width, int height, int nc) {
    int id_X = blockIdx.x * blockDim.x + threadIdx.x;
    int id_Y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_X = gridDim.x * blockDim.x;
    int offset_Y = gridDim.y * blockDim.y;

    for (int row = id_Y; row < height; row += offset_Y) {
        for (int col = id_X; col < width; col += offset_X) {
            int timed = 0;
            double max = get_current_pos(&dot, 0);
            uchar4 dot = photo_dots[row * width + col];
            for (int i = 1; i < nc; i++) {
                double next = get_current_pos(&dot, i);
                if (next > max) {
                    max = next;
                    timed = i;
                }
            }
            photo_dots[row * width + col].w = (unsigned char)timed;
        }
    }
}


void reversation(matrix &matrix_vals) {
    double minor_part11, minor_part12, minor_part21, minor_part22, minor_part31, minor_part32;

    minor_part11 = matrix_vals.dots[2][2] * matrix_vals.dots[1][1];
    minor_part12 = matrix_vals.dots[1][2] * matrix_vals.dots[2][1];
    minor_part21 = matrix_vals.dots[2][2] * matrix_vals.dots[1][0];
    minor_part22 = matrix_vals.dots[1][2] * matrix_vals.dots[2][0];
    minor_part31 = matrix_vals.dots[2][1] * matrix_vals.dots[1][0];
    minor_part32 = matrix_vals.dots[1][1] * matrix_vals.dots[2][0];

    double minor1 =  (minor_part11 - minor_part12);
    double minor2 =  (minor_part22 - minor_part21);
    double minor3 =  (minor_part31 - minor_part32);

    double minor_part41, minor_part42, minor_part51, minor_part52, minor_part61, minor_part62;

    double minor4 = minor_part42 - minor_part41;
    double minor5 = minor_part51 - minor_part52;
    double minor6 = minor_part62 - minor_part61;

    double minor_part71, minor_part72, minor_part81, minor_part82, minor_part91, minor_part92;

    double minor7 = minor_part71 - minor_part72;
    double minor8 = minor_part82 - minor_part81,;
    double minor9 = minor_part91 - minor_part92;

    double minor[3][3] = {{minor1, minor4, minor7},{minor2, minor5, minor8},{minor3, minor6, minor9}};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            matrix_vals.dots[i][j] = minor[i][j] / (matrix_vals.dots[0][2] * minor3 + matrix_vals.dots[0][0] * minor1 - ((-1) * matrix_vals.dots[0][1] * (minor2)));
        }
    }
}


int main() {
    string output_file_name, input_file_name;
    int nc, width, height;

    cin >> input_file_name;
    cin >> output_file_name;
    cin >> nc;

    vector<vector<position>> input_photo;
    input_photo.resize(32);

    int class_pixels_number;
    for (int row = 0; row < nc; row++) {
        cin >> class_pixels_number;
        input_photo[row].resize(class_pixels_number);
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


    start(photo_dots, input_photo, nc, width);
    CSC(cudaMemcpyToSymbol(vec, void_vec, 32 * sizeof(vec_st)));
    CSC(cudaMemcpyToSymbol(dots, void_matr, 32 * sizeof(matrix)));

    uchar4* output_dots;
    CSC(cudaMalloc(&output_dots, sizeof(uchar4) * width * height));
    CSC(cudaMemcpy(output_dots, photo_dots, sizeof(uchar4) * width * height, cudaMemcpyHostToDevice));

    kernel<<<dim3(32, 32), dim3(32, 32)>>>(output_dots, width, height, nc);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(photo_dots, output_dots, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

    CSC(cudaFree(output_dots));

    fwrite(&width, sizeof(int), 1, output_file);
    fwrite(&height, sizeof(int), 1, output_file);
    fwrite(photo_dots, sizeof(uchar4), width * height, output_file);
    fclose(output_file);

    free(photo_dots);

    return 0;
}