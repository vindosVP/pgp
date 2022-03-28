#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
using namespace std;

#define _i(i, j, k) (((k) + 1) * (nX + 2) * (nY + 2) + ((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nb_X * nb_Y + (j) * nb_X + (i))


__global__ void init_block(double* data, int nX, int nY, int nZ, double init_v) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz; k < nZ; k+=offsetz) {
    	for (int j = idy; j < nY; j+=offsety) {
    		for (int i = idx; i < nX; i+=offsetx) {
    			data[_i(i, j, k)] = init_v;
    		}
    	}
    }
}

__global__ void up_down_bc(double* data, int nX, int nY, double bc, int z_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < nY; j+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		data[_i(i, j, z_Idx)] = bc;
    	}
    }
}

__global__ void front_back_bc(double* data, int nX, int nY, int nZ, double bc, int y_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		data[_i(i, y_Idx, k)] = bc;
    	}
    }
}

__global__ void left_right_bc(double* data, int nX, int nY, int nZ, double bc, int x_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int j = idx; j < nY; j+=offsetx) {
    		data[_i(x_Idx, j, k)] = bc;
    	}
    }
}


__global__ void send_left_right(double* buf, double* data, int nX, int nY, int nZ, int x_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int j = idx; j < nY; j+=offsetx) {
    		buf[k * nY + j] = data[_i(x_Idx, j, k)];
    	}
    }
}

__global__ void get_left_right(double* buf, double* data, int nX, int nY, int nZ, int x_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int j = idx; j < nY; j+=offsetx) {
    		data[_i(x_Idx, j, k)] = buf[k * nY + j];
    	}
    }
}

__global__ void send_front_back(double* buf, double* data, int nX, int nY, int nZ, int y_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		buf[k * nX + i] = data[_i(i, y_Idx, k)];
    	}
    }
}

__global__ void get_front_back(double* buf, double* data, int nX, int nY, int nZ, int y_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy; k < nZ; k+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		data[_i(i, y_Idx, k)] = buf[k * nX + i];
    	}
    }
}

__global__ void send_up_down(double* buf, double* data, int nX, int nY, int z_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < nY; j+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		buf[j * nX + i] = data[_i(i, j, z_Idx)];
    	}
    }
}

__global__ void get_up_down(double* buf, double* data, int nX, int nY, int z_Idx) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < nY; j+=offsety) {
    	for (int i = idx; i < nX; i+=offsetx) {
    		data[_i(i, j, z_Idx)] = buf[j * nX + i];
    	}
    }
}

__global__ void next_element(double* data, double* next, int nX, int nY, int nZ, double hx, double hy, double hz) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz; k < nZ; k+=offsetz) {
    	for (int j = idy; j < nY; j+=offsety) {
    		for (int i = idx; i < nX; i+=offsetx) {
    			next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
						(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
						(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
    		}
    	}
    }
}

__global__ void mistake(double* data, double* next, int nX, int nY, int nZ) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int offsetz = blockDim.z * gridDim.z;

    for (int k = idz - 1; k <= nZ; k+=offsetz) {
    	for (int j = idy - 1; j <= nY; j+=offsety) {
    		for (int i = idx - 1; i <= nX; i+=offsetx) {
    			 if ((i == -1) || (j == -1) || (k == -1) || (i == nX) || (j == nY) || (k == nZ)) {
                    data[_i(i, j, k)] = 0.0;
                } else {
                    data[_i(i, j, k)] = fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
    			}
    		}	
    	}
    }
}


int main(int argc, char* argv[]){
	int b_res;
	int gpu_num;
	int nums = 3;
	char output_file_name[100];
	int ib, jb, kb, nb_X, nb_Y, nb_Z, nX, nY, nZ;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, init_v, epsilon, diff;
	double *gpu_data, *temp, *gpu_next, *buff;


	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);
	cudaGetDeviceCount(&gpu_num);
	cudaSetDevice(id % gpu_num);

	if (id == 0) {																			  //праметры расчета
		cin >> nb_X >> nb_Y >> nb_Z;														  //размер блока
		cin >> nX >> nY >> nZ;     															  // размер сетки
		cin >> output_file_name;															  // имя файла в который пишем результат
		cin >> epsilon;																		  // точность
		cin >> lx >> ly >> lz;																  // размеры области
		cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back; 				  // граничные условия
		cin >> init_v;																		  // начальное значение
	}


//----------------------------------------------------------------------------------// Распространение данных на все процессы
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nY, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nZ, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nb_Z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&init_v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(output_file_name, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
	diff = epsilon + 1;

//---------------------------------------------------------------------------------// Вычисление индексов(переход к трехмерной индексации)
	
	kb = id / (nb_X * nb_Y);
	jb = id % (nb_X * nb_Y) / nb_X;
	ib = id % (nb_X * nb_Y) % nb_X;
	
	hx = lx / (nX * nb_X);
	hy = ly / (nY * nb_Y);
	hz = lz / (nZ * nb_Z);

//---------------------------------------------------------------------------------// Выделение памяти (n+2 из-за дополнительных ячеек)

	cudaMalloc(&gpu_data, sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));
	cudaMalloc(&gpu_next, sizeof(double) * (nY + 2) * (nY + 2) * (nZ + 2));
	buff = (double*)malloc(sizeof(double) * max(nX, nY) * max(nY, nZ));
	double* output = (double *)malloc(sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));

//---------------------------------------------------------------------------------// Определение рзмер буфера

	MPI_Pack_size(max(nX, nY) * max(nY, nZ), MPI_DOUBLE, MPI_COMM_WORLD, &b_res);
	b_res = 6 * (b_res + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(b_res);
	double* gpu_buff;
	MPI_Buffer_attach(buffer, b_res);
	cudaMalloc(&gpu_buff, sizeof(double) * max(nX, nY) * max(nY, nZ));

//---------------------------------------------------------------------------------// Инициализаця блока	
	
	init_block<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(gpu_data, nX, nY, nZ, init_v);


//---------------------------------------------------------------------------------// Отправка данных
	while (true) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nb_X) {
			send_left_right<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, nX-1);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nY * nZ, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nY * nZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nb_Y) {
			send_front_back<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, nY-1);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nX * nZ, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < nb_Z) {
			send_up_down<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ-1);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nX * nY, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			send_left_right<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, 0);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nY * nZ, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nY * nZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			send_front_back<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, 0);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nX * nZ, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			send_up_down<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, 0);

			cudaMemcpy(buff, gpu_buff, sizeof(double) * nX * nY, cudaMemcpyDeviceToHost);
			MPI_Bsend(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

//---------------------------------------------------------------------------------// Прием данных

		if (ib > 0) {
			MPI_Recv(buff, nY * nZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nY * nZ, cudaMemcpyHostToDevice);
			get_left_right<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, -1);

		} else {
			left_right_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, nZ, bc_left, -1);

		}

		if (jb > 0) {
			MPI_Recv(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nX * nZ, cudaMemcpyHostToDevice);
			get_front_back<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, -1);

		} else {
			front_back_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, nZ, bc_front, -1);

		}

		if (kb > 0) {
			MPI_Recv(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nX * nY, cudaMemcpyHostToDevice);
			get_up_down<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, -1);

		} else {
			up_down_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, bc_down, -1);

		}

		if (ib + 1 < nb_X) {
			MPI_Recv(buff, nY * nZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nY * nZ, cudaMemcpyHostToDevice);
			get_left_right<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, nX);

		} else {
			left_right_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, nZ, bc_right, nX);

		}

		if (jb + 1 < nb_Y) {
			MPI_Recv(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nX * nZ, cudaMemcpyHostToDevice);
			get_front_back<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ, nY);

		} else {
			front_back_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, nZ, bc_back, nY);

		}

		if (kb + 1 < nb_Z) {
			MPI_Recv(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			cudaMemcpy(gpu_buff, buff, sizeof(double) * nX * nY, cudaMemcpyHostToDevice);
			get_up_down<<<dim3(32,32), dim3(32,32)>>>(gpu_buff, gpu_data, nX, nY, nZ);

		} else {
			up_down_bc<<<dim3(32,32), dim3(32,32)>>>(gpu_data, nX, nY, bc_up, nZ);

		}

		MPI_Barrier(MPI_COMM_WORLD);
//---------------------------------------------------------------------------------// Обмен и Allreduce
		diff = 0.0;

		next_element<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(gpu_data, gpu_next, nX, nY, nZ, hx, hy, hz);

		mistake<<<dim3(8, 8, 8), dim3(32, 4, 4)>>>(gpu_data, gpu_next, nX, nY, nZ);

		thrust::device_ptr<double> diff_part = thrust::device_pointer_cast(gpu_data);
        thrust::device_ptr<double> diff_now = thrust::max_element(diff_part, diff_part + (nX + 2) * (nY + 2) * (nZ + 2));
        diff =  *diff_now;
		MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
		temp = gpu_next;
		gpu_next = gpu_data;
		gpu_data = temp;
		
		if(diff <= epsilon){
			break;
		}
	}



//---------------------------------------------------------------------------------// Вывод днных
	cudaMemcpy(output, gpu_data, sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2), cudaMemcpyDeviceToHost);
	


	//--------------------------------------------------------------------------------------------------------------------------------//

	int n_size = 14;	
	char* out = (char*)malloc(sizeof(char) * nX * nY * nZ * n_size);
	memset(out, ' ', sizeof(char) * nX * nY * nZ *n_size);

	for (int k = 0; k < nZ; k++) {
		for (int j = 0; j < nY; j++) {
			for (int i = 0; i < nX; i++) {
				sprintf(out + (k * nX * nY + j * nX + i) * n_size, "%.6e", output[_i(i, j, k)]);
			}
		}
	}
	
	for (int i = 0; i < nX * nY * nZ * n_size; i++) {
		if (out[i] == '\0') {
			out[i] = ' ';
		}
	}
	
	MPI_File output_file;
	MPI_Datatype out_type;
	int sizes[] = {nZ * nb_Z, nY * nb_Y, nX * nb_X * n_size};
	int subsizes[] = {nZ, nY, nX * n_size};
	int starts[] = {id / (nb_X * nb_Y) * nZ, id % (nb_X * nb_Y) / nb_X * nY,  id % (nb_X * nb_Y) % nb_X * nX * n_size};
	MPI_Type_create_subarray(nums, sizes, subsizes, starts, MPI_ORDER_C, MPI_CHAR, &out_type);
	MPI_Type_commit(&out_type);

	MPI_File_delete(output_file_name, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, output_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
	MPI_File_set_view(output_file, 0, MPI_CHAR, out_type, "native", MPI_INFO_NULL);
	MPI_File_write_all(output_file, out, nX * nY * nZ * n_size, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&output_file);

	MPI_Type_free(&out_type);
	MPI_Finalize();
	cudaFree(gpu_next);
	cudaFree(gpu_buff);
	cudaFree(gpu_data);
	return 0;	
}