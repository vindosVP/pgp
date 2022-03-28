#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"
#include <omp.h>
using namespace std;

#define _i(i, j, k) (((k) + 1) * (nX + 2) * (nY + 2) + ((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nb_X * nb_Y + (j) * nb_X + (i))

int main(int argc, char* argv[]){
	int i,j,k;
	int pgl = 1;
	int nums = 3;
	int n_size = 14;
	char output_file_name[100];
	int ib, jb, kb, nb_X, nb_Y, nb_Z, nX, nY, nZ;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, init_v, epsilon, diff;
	double *data, *temp, *next;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);

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

//---------------------------------------------------------------------------------// Вычисление индексов(переход к трехмерной индексации)
	
	kb = id / (nb_X * nb_Y);
	jb = id % (nb_X * nb_Y) / nb_X;
	ib = id % (nb_X * nb_Y) % nb_X;
	
	hx = lx / (nX * nb_X);
	hy = ly / (nY * nb_Y);
	hz = lz / (nZ * nb_Z);
	diff = epsilon + 1;

//---------------------------------------------------------------------------------// initing type parts

	int main_part[] = {nZ + 2, nY + 2, nX + 2};
	int minor_part_left_right[] = {nZ, nY, 1};
	int minor_part_front_back[] = {nZ, 1, nX};
	int minor_part_up_down[] = {1, nY, nX};
	int init_full[] = {1, 1, 1};
	int init_of_idZ[] = {nZ, 1, 1};
	int init_of_idY[] = {1, nY, 1};
	int init_of_idX[] = {1, 1, nX};
	int init_beg_X[] = {1, 1, 0};
	int init_beg_Y[] = {1, 0, 1};
	int init_beg_Z[] = {0, 1, 1};
	int init_x[] = {1, 1, nX + 1};
	int init_y[] = {1, nY + 1, 1};
	int init_z[] = {nZ + 1, 1, 1};
	int main_part_file[] = {nZ * nb_Z, nY * nb_Y, nX * nb_X * n_size};
	int minor_part_file[] = {nZ, nY, nX * n_size};
	int init_file[] = {id / (nb_X * nb_Y) * nZ, id % (nb_X * nb_Y) / nb_X * nY,  id % (nb_X * nb_Y) % nb_X * nX * n_size};

//---------------------------------------------------------------------------------// initing types

	//send
	MPI_Datatype send_left;
	MPI_Datatype send_right;
	MPI_Datatype send_front;
	MPI_Datatype send_back;
	MPI_Datatype send_up;
	MPI_Datatype send_down;


	//recieve
	MPI_Datatype recieve_left;
	MPI_Datatype recieve_right;
	MPI_Datatype recieve_front;
	MPI_Datatype recieve_back;
	MPI_Datatype recieve_up;
	MPI_Datatype recieve_down;

	//file
	MPI_Datatype out_type;
	MPI_File output_file;


//---------------------------------------------------------------------------------// creating arrays

	//send
	MPI_Type_create_subarray(nums, main_part, minor_part_left_right, init_full, MPI_ORDER_C, MPI_DOUBLE, &send_left);
	MPI_Type_create_subarray(nums, main_part, minor_part_left_right, init_of_idX, MPI_ORDER_C, MPI_DOUBLE, &send_right);
	MPI_Type_create_subarray(nums, main_part, minor_part_front_back, init_full, MPI_ORDER_C, MPI_DOUBLE, &send_front);
	MPI_Type_create_subarray(nums, main_part, minor_part_front_back, init_of_idY, MPI_ORDER_C, MPI_DOUBLE, &send_back);
	MPI_Type_create_subarray(nums, main_part, minor_part_up_down, init_of_idZ, MPI_ORDER_C, MPI_DOUBLE, &send_up);
	MPI_Type_create_subarray(nums, main_part, minor_part_up_down, init_full, MPI_ORDER_C, MPI_DOUBLE, &send_down);


	//recieve
	MPI_Type_create_subarray(nums, main_part, minor_part_left_right, init_beg_X, MPI_ORDER_C, MPI_DOUBLE, &recieve_left);
	MPI_Type_create_subarray(nums, main_part, minor_part_left_right, init_x, MPI_ORDER_C, MPI_DOUBLE, &recieve_right);
	MPI_Type_create_subarray(nums, main_part, minor_part_front_back, init_beg_Y, MPI_ORDER_C, MPI_DOUBLE, &recieve_front);
	MPI_Type_create_subarray(nums, main_part, minor_part_front_back, init_y, MPI_ORDER_C, MPI_DOUBLE, &recieve_back);
	MPI_Type_create_subarray(nums, main_part, minor_part_up_down, init_z, MPI_ORDER_C, MPI_DOUBLE, &recieve_up);
	MPI_Type_create_subarray(nums, main_part, minor_part_up_down, init_beg_Z, MPI_ORDER_C, MPI_DOUBLE, &recieve_down);

	//file
	MPI_Type_create_subarray(nums, main_part_file, minor_part_file, init_file, MPI_ORDER_C, MPI_CHAR, &out_type);

//---------------------------------------------------------------------------------// commiting types

	
	//send
	MPI_Type_commit(&send_left);
	MPI_Type_commit(&send_right);
	MPI_Type_commit(&send_front);
	MPI_Type_commit(&send_back);
	MPI_Type_commit(&send_up);
	MPI_Type_commit(&send_down);


	//recieve
	MPI_Type_commit(&recieve_left);
	MPI_Type_commit(&recieve_right);
	MPI_Type_commit(&recieve_front);
	MPI_Type_commit(&recieve_back);
	MPI_Type_commit(&recieve_up);
	MPI_Type_commit(&recieve_down);

	//file
	MPI_Type_commit(&out_type);

//---------------------------------------------------------------------------------// Выделение памяти (n+2 из-за дополнительных ячеек)

	data = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));
	next = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));
	char* text = (char*)malloc(sizeof(char) * nX * nY * nZ * n_size);
	memset(text, ' ', sizeof(char) * nX * nY * nZ *n_size);
	MPI_File_delete(output_file_name, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, output_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
	MPI_File_set_view(output_file, 0, MPI_CHAR, out_type, "native", MPI_INFO_NULL);


//---------------------------------------------------------------------------------// Инициализаця блока	
	
for (i = 0; i < nX; i++) {					
		for (j = 0; j < nY; j++) {
			for (k = 0; k < nZ; k++) {
				data[_i(i, j, k)] = init_v;
			}
		}
	}

	for (j = 0; j < nY; j++) {
		for (k = 0; k < nZ; k++) {
			data[_i(-1, j, k)] = bc_left;
			next[_i(-1, j, k)] = bc_left;
		}
	}

	for (i = 0; i < nX; i++) {
		for (k = 0; k < nZ; k++) {
			data[_i(i, -1, k)] = bc_front;
			next[_i(i, -1, k)] = bc_front;
		}
	}

	for (i = 0; i < nX; i++) {
		for (j = 0; j < nY; j++) {
			data[_i(i, j, -1)] = bc_down;
			next[_i(i, j, -1)] = bc_down;
		}
	}

	for (j = 0; j < nY; j++) {
		for (k = 0; k < nZ; k++) {
			data[_i(nX, j, k)] = bc_right;
			next[_i(nX, j, k)] = bc_right;
		}
	}

	for (i = 0; i < nX; i++) {
		for (k = 0; k < nZ; k++) {
			data[_i(i, nY, k)] = bc_back;
			next[_i(i, nY, k)] = bc_back;
		}
	}

	for (i = 0; i < nX; i++) {
		for (j = 0; j < nY; j++) {
			data[_i(i, j, nZ)] = bc_up;
			next[_i(i, j, nZ)] = bc_up;
		}
	}



//---------------------------------------------------------------------------------// Отправка данных
	while (diff >= epsilon) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nb_X) MPI_Bsend(data, pgl, send_right, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb + 1 < nb_Y) MPI_Bsend(data, pgl, send_back, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		if (kb + 1 < nb_Z) MPI_Bsend(data, pgl, send_up, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		if (ib > 0) 	   MPI_Bsend(data, pgl, send_left, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		if (jb > 0)        MPI_Bsend(data, pgl, send_front, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		if (kb > 0)        MPI_Bsend(data, pgl, send_down, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		
//---------------------------------------------------------------------------------// Прием данных

		if (ib > 0)        MPI_Recv(data, pgl, recieve_left, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb > 0)        MPI_Recv(data, pgl, recieve_front, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
		if (kb > 0)        MPI_Recv(data, pgl, recieve_down, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
		if (ib + 1 < nb_X) MPI_Recv(data, pgl, recieve_right, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
		if (jb + 1 < nb_Y) MPI_Recv(data, pgl, recieve_back, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
		if (kb + 1 < nb_Z) MPI_Recv(data, pgl, recieve_up, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
		
		MPI_Barrier(MPI_COMM_WORLD);
//---------------------------------------------------------------------------------// Обмен и Allreduce
		diff = 0.0;
		#pragma omp parallel shared(data, next) reduction(max:diff)
		{	
			int offsetX = omp_get_num_threads();
			int idX = omp_get_thread_num();

			for (int iter = idX; iter < nX * nY * nZ; iter += offsetX) {
				 int i = iter % nX;
				 int k = iter / (nX * nY);
				 int j = iter % (nX * nY) / nX;

				next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
					(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
					(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
					(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

				diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
			}
		}

		temp = next;
		next = data;
		data = temp;

		MPI_Allreduce(MPI_IN_PLACE, &diff, pgl, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
	}


//---------------------------------------------------------------------------------// Вывод днных
	for (k = 0; k < nZ; k++) 
		for (j = 0; j < nY; j++) 
			for (i = 0; i < nX; i++) 
				sprintf(text + (k * nX * nY + j * nX + i) * n_size, "%.6e", data[_i(i, j, k)]);
			
	for (int i = 0; i < nX * nY * nZ * n_size; i++) 
		if (text[i] == '\0') text[i] = ' ';
		
	MPI_File_write_all(output_file, text, nX * nY * nZ * n_size, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&output_file);

	MPI_Type_free(&out_type);
	MPI_Finalize();
	return 0;	
}