#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include "mpi.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
using namespace std;

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

#define _i(i, j) (((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j) ((j) * nb_X + (i))


__global__ void left_right_bc(double *data, int ind_x, double bc, int nX, int nY) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.x * gridDim.x;

    for (int j = idy; j < nY; j += offsety) {
        data[_i(ind_x, j)] = bc;
    }
}

__global__ void up_down_bc(double *data, int ind_y, double bc, int nX, int nY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nX; i += offsetx) {
        data[_i(i, ind_y)] = bc;
    }
}

__global__ void send_left_right(double *data, double *buffer, int ind_x, int nX, int nY) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.x * gridDim.x;

    for (int j = idy; j < nY; j += offsety) {
        buffer[j] = data[_i(ind_x, j)];
    }
}

__global__ void recieve_left_right(double *data, double *buffer, int ind_x, int nX, int nY) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    int offsety = blockDim.x * gridDim.x;

    for (int j = idy; j < nY; j += offsety) {
        data[_i(ind_x, j)] = buffer[j];
    }
}

__global__ void send_up_down(double *data, double *buffer, int ind_y, int nX, int nY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nX; i += offsetx) {
        buffer[i] = data[_i(i, ind_y)];
    }
}

__global__ void recieve_up_down(double *data, double *buffer, int ind_y, int nX, int nY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nX; i += offsetx) {
        data[_i(i, ind_y)] = buffer[i];
    }
}

__global__ void next_element(double* data, double* next, int nX, int nY, double hx, double hy) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    	for (int j = idy; j < nY; j+=offsety) {
    		for (int i = idx; i < nX; i+=offsetx) {
    			next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
                                    (data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) /
                                 (1.0 / (hx * hx) + 1.0 / (hy * hy));
    		}
    	}
    }


__global__ void mistake(double* data, double* next, int nX, int nY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    	for (int j = idy - 1; j <= nY; j+=offsety) {
    		for (int i = idx - 1; i <= nX; i+=offsetx) {
    			 data[_i(i, j)] = ((i != -1) && (j != -1) && (i != nX) && (j != nY)) * fabs(next[_i(i, j)] - data[_i(i, j)]);
    		}	
    	}
    }



int main(int argc, char* argv[]){
	int n_size = 16;
	int gpu_num;
	char output_file_name[100];
	int ib, jb, nb_X, nb_Y, nX, nY;
	int id, numproc;
	double lx, ly, hx, hy, bc_down, bc_up, bc_left, bc_right,init_v, epsilon, diff;
	double *gpu_data, *temp, *gpu_next, *buff;


	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);
	cudaGetDeviceCount(&gpu_num);
	cudaSetDevice(id % gpu_num);

	if (id == 0) {																			  //праметры расчета
		cin >> nb_X >> nb_Y;														  //размер блока
		cin >> nX >> nY;     															  // размер сетки
		cin >> output_file_name;															  // имя файла в который пишем результат
		cin >> epsilon;																		  // точность
		cin >> lx >> ly;																  // размеры области
		cin >> bc_left >> bc_right >> bc_down >> bc_up; 				  // граничные условия
		cin >> init_v;																		  // начальное значение
	}


//----------------------------------------------------------------------------------// Распространение данных на все процессы
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nY, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&init_v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(output_file_name, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
	diff = epsilon + 1;

//---------------------------------------------------------------------------------// Вычисление индексов(переход к трехмерной индексации)
	

	jb = id / nb_X;
	ib = id % nb_X;
	
	hx = lx / (nX * nb_X);
	hy = ly / (nY * nb_Y);


//---------------------------------------------------------------------------------// Выделение памяти (n+2 из-за дополнительных ячеек)

	cudaMalloc(&gpu_data, sizeof(double) * (nX + 2) * (nY + 2));
	cudaMalloc(&gpu_next, sizeof(double) * (nY + 2) * (nY + 2));
	buff = (double*)malloc(sizeof(double) * max(nX, nY) + 2);
	double* output = (double *)malloc(sizeof(double) * (nX + 2) * (nY + 2));
    unsigned long long size = n_size * nX * nY;

//---------------------------------------------------------------------------------// Определение рзмер буфера

    int b_size = max(nX, nY) + 2;
    buff = (double*)malloc(sizeof(double) * b_size);
    double *gpu_buff;
    CSC(cudaMalloc(&gpu_buff, sizeof(double) * b_size));
    int buffer_size;
    MPI_Pack_size(max(nX, nY) + 2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);
    char *text = (char*) malloc(sizeof(char) * size);  
    memset(text, ' ', sizeof(char) * size);

//---------------------------------------------------------------------------------// Инициализаця блока	
	for (int i = 0; i < nX; i++) {
        for (int j = 0; j < nY; j++) {
            data[_i(i, j)] = init_v;
        }
    }
    cudaMemcpy(gpu_data, data, sizeof(double) * (nX + 2) * (nY + 2), cudaMemcpyHostToDevice)

//---------------------------------------------------------------------------------// Отправка данных
	while (true) {
		MPI_Barrier(MPI_COMM_WORLD);

		left_right_bc<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, -1, bc_left, nX, nY);
        left_right_bc<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, nX, bc_right, nX, nY);
        up_down_bc<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, -1, bc_down, nX, nY);
        up_down_bc<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, nY, bc_up, nX, nY);

		if (ib + 1 < nb_X) {
            send_left_right<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, nX - 1, nX, nY);
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * b_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(buff, nY, MPI_DOUBLE, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < nb_Y) {
            send_up_down<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, nY - 1, nX, nY);
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * b_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(buff, nX, MPI_DOUBLE, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            send_left_right<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, 0, nX, nY);
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * b_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(buff, nY, MPI_DOUBLE, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            send_up_down<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, 0, nX, nY);
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * b_size, cudaMemcpyDeviceToHost));
            MPI_Bsend(buff, nX, MPI_DOUBLE, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
        }

//---------------------------------------------------------------------------------// Прием данных

		if (ib > 0) {
            MPI_Recv(buff, nY, MPI_DOUBLE, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * b_size, cudaMemcpyHostToDevice));
            recieve_left_right<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, -1, nX, nY);
        }

        if (jb > 0) {
            MPI_Recv(buff, nX, MPI_DOUBLE, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * b_size, cudaMemcpyHostToDevice));
            recieve_up_down<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, -1, nX, nY);
        }

        if (ib + 1 < nb_X) {
            MPI_Recv(buff, nY, MPI_DOUBLE, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * b_size, cudaMemcpyHostToDevice));
            recieve_left_right<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, nX, nX, nY);
        }

        if (jb + 1 < nb_Y) {
            MPI_Recv(buff, nX, MPI_DOUBLE, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * b_size, cudaMemcpyHostToDevice));
            recieve_up_down<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_buff, nY, nX, nY);
        }

		MPI_Barrier(MPI_COMM_WORLD);
//---------------------------------------------------------------------------------// Обмен и Allreduce
		diff = 0.0;

		next_element<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_next, nX, nY, hx, hy);

		mistake<<<dim3(16, 16), dim3(16, 16)>>>(gpu_data, gpu_next, nX, nY);

		thrust::device_ptr<double> diff_part = thrust::device_pointer_cast(gpu_data);
        thrust::device_ptr<double> diff_now = thrust::max_element(diff_part, diff_part + (nX + 2) * (nY + 2));
        diff =  *diff_now;
		
		temp = gpu_next;
		gpu_next = gpu_data;
		gpu_data = temp;
		MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if(diff < epsilon){
			break;
		}
	}



//---------------------------------------------------------------------------------// Вывод днных
	cudaMemcpy(output, gpu_data, sizeof(double) * (nX + 2) * (nY + 2), cudaMemcpyDeviceToHost);
	


	//--------------------------------------------------------------------------------------------------------------------------------//
	for (int i = 0; i < nY; i++) 
        for (int j = 0; j < nX; j++) 
            sprintf(text + (i * nX + j) * n_size, "%.7e", output[_i(j, i)]);
        
    
    for (size_t i = 0; i < n_size * nX * nY; i++) 
        if (text[i] == '\0') text[i] = ' ';
    
    MPI_Datatype out_type;
    int out_type_part3 = n_size * nX * nb_X;
    int out_type_part2 = n_size * nX;
    int out_type_part1 = nY;
    int aint_part1 = nX * ib;
    int aint_part2 = nY * jb;
    MPI_Type_create_hvector(out_type_part1, out_type_part2, out_type_part3, MPI_CHAR, &out_type);
    MPI_Type_commit(&out_type);

    MPI_Aint format = out_type_part3 * aint_part2 + n_size * aint_part1;

    MPI_File output_file;
    MPI_File_delete(output_file_name, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_set_view(output_file, format, MPI_CHAR, out_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(output_file, text, size, MPI_CHAR, &status);
    MPI_File_close(&output_file);
    
    MPI_Finalize();
    cudaFree(gpu_next);
    cudaFree(gpu_buff);
    cudaFree(gpu_data);
	return 0;	
}