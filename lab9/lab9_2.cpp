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

#define _i(i, j) (((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j) ((j) * nb_X + (i))

int main(int argc, char* argv[]){
	char output_file_name[100];
	int ib, jb, nb_X, nb_Y, nX, nY;
	int n_size = 16;
	int id, numproc;
	double lx, ly, hx, hy, bc_down, bc_up, bc_left, bc_right, init_v, epsilon, diff;
	double *next, *data, *temp;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {																			  //праметры расчета
		cin >> nb_X >> nb_Y;														          //размер блока
		cin >> nX >> nY;     															      // размер сетки
		cin >> output_file_name;															  // имя файла в который пишем результат
		cin >> epsilon;																		  // точность
		cin >> lx >> ly;																      // размеры области
		cin >> bc_left >> bc_right >> bc_down >> bc_up; 				                      // граничные условия
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

//---------------------------------------------------------------------------------// Вычисление индексов(переход к трехмерной индексации)
	
	jb = id / nb_X;
	ib = id % nb_X; 
	
	hx = lx / (nX * nb_X);
	hy = ly / (nY * nb_Y);
	diff = epsilon + 1;


//---------------------------------------------------------------------------------// initing type parts
	
	int sendrec_left_right_part = nY + 2;
	int sendrec_up_down_part = nX + 2;
	int out_type_minot_part1 = nY;
	int out_type_minot_part2 = n_size * nX;
	int out_type_minot_part3 = n_size * nX * nb_X;
	unsigned int size = n_size * nX * nY;
    MPI_Aint format = n_size * nX * nb_X * nY * jb + n_size * nX * ib;

//---------------------------------------------------------------------------------// initing types

	//sendrecie
	MPI_Datatype sendrec_left_right;
	MPI_Datatype sendrec_up_down;

	//file
	MPI_Datatype out_type;
	MPI_File output_file;


//---------------------------------------------------------------------------------// creating arrays

	//sendrecieve
    MPI_Type_vector(sendrec_left_right_part, 1, sendrec_up_down_part, MPI_DOUBLE, &sendrec_left_right);
    MPI_Type_vector(1, sendrec_up_down_part, 0, MPI_DOUBLE, &sendrec_up_down);


	//file
	MPI_Type_create_hvector(out_type_minot_part1, out_type_minot_part2, out_type_minot_part3, MPI_CHAR, &out_type);

//---------------------------------------------------------------------------------// commiting types
	//sendrec
 	MPI_Type_commit(&sendrec_left_right);
 	MPI_Type_commit(&sendrec_up_down);

	//file
 	MPI_Type_commit(&out_type);
 	MPI_File_delete(output_file_name, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_set_view(output_file, format, MPI_CHAR, out_type, "native", MPI_INFO_NULL);

//---------------------------------------------------------------------------------// Выделение памяти (n+2 из-за дополнительных ячеек)

	data = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2));
	next = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2));
    int buffer_size;
    MPI_Pack_size((max(nX, nY) + 2), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);
	char* text = (char*)malloc(sizeof(char) * size);
	memset(text, ' ', sizeof(char) * n_size * nX * nY);


//---------------------------------------------------------------------------------// Инициализаця блока	
	int i,j;
	#pragma omp parallel for private(i, j) shared(data)
	for (i = 0; i < nX; i++) {					
		for (j = 0; j < nY; j++) {
				data[_i(i, j)] = init_v;
			}
		}

	#pragma omp parallel for private(j) shared(next, data)
		for (j = 0; j < nY; j++) {
			data[_i(-1, j)] = bc_left;
			next[_i(-1, j)] = bc_left;
		}

	#pragma omp parallel for private(i) shared(next, data)
		for (i = 0; i < nX; i++) {
			data[_i(i, -1)] = bc_down;
			next[_i(i, -1)] = bc_down;
		}

	#pragma omp parallel for private(j) shared(next, data)
		for (j= 0; j < nY; j++) {
			data[_i(nX, j)] = bc_right;
			next[_i(nX, j)] = bc_right;
		}

	#pragma omp parallel for private(i) shared(next, data)		
		for (i = 0; i < nX; i++) {
			data[_i(i, nY)] = bc_up;
			next[_i(i, nY)] = bc_up;
		}
    


//---------------------------------------------------------------------------------// Отправка данных
	while (true) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nb_X)MPI_Bsend(data + nX, 1, sendrec_left_right, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
		if (jb + 1 < nb_Y)MPI_Bsend(data + (nX + 2) * nY, 1, sendrec_up_down, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
		if (ib > 0)       MPI_Bsend(data + 1, 1, sendrec_left_right, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
		if (jb > 0)       MPI_Bsend(data + nX + 2, 1, sendrec_up_down, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
		
//---------------------------------------------------------------------------------// Прием данных

		if (ib > 0)       MPI_Recv(data, 1, sendrec_left_right, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
		if (jb > 0)       MPI_Recv(data, 1, sendrec_up_down, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
		if (ib + 1 < nb_X)MPI_Recv(data + nX + 1, 1, sendrec_left_right, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
		if (jb + 1 < nb_Y)MPI_Recv(data + (nX + 2) * (nY - 1), 1, sendrec_up_down, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);

		MPI_Barrier(MPI_COMM_WORLD);
//---------------------------------------------------------------------------------// Обмен и Allreduce
		diff = 0.0;
		#pragma omp parallel for private(i, j) shared(data, next) reduction(max:diff)
		for (int i = 0; i < nX; i++){
			for (int j = 0; j < nY; j++){
				next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
                        (data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) /
                        (1.0 / (hx * hx) + 1.0 / (hy * hy));

				diff = fmax(diff, fabs(next[_i(i, j)] - data[_i(i, j)]));
			}
		}
			
		temp = next;
		next = data;
		data = temp;

		MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
		if (diff < epsilon){
			break;
		}
	}
		
	


//---------------------------------------------------------------------------------// Вывод днных
    
    for (int i = 0; i < nY; i++) 
        for (int j = 0; j < nX; j++)
            sprintf(text + (i * nX + j) * n_size, "%.7e", data[_i(j, i)]);
        
    
    for (size_t i = 0; i < size; i++)
        if (text[i] == '\0') text[i] = ' ';
    






    MPI_File_write_all(output_file, text, size, MPI_CHAR, &status);
    MPI_File_close(&output_file);


	MPI_Finalize();
	return 0;	
}   