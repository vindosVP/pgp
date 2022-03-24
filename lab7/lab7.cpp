#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <time.h>
#include "mpi.h"
using namespace std;

#define _i(i, j, k) (((k) + 1) * (nX + 2) * (nY + 2) + ((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nb_X * nb_Y + (j) * nb_X + (i))

int main(int argc, char* argv[]){
	int buffer_size;
	string output_file_name;
	int ib, jb, kb, nb_X, nb_Y, nb_Z, nX, nY, nZ;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, init_v, epsilon, diff;
	double *data, *temp, *next, *buff;
	int i,j,k;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {									//праметры расчета
		cin >> nb_X >> nb_Y >> nb_Z;	//размер блока
		cin >> nX >> nY >> nZ;      //размер сетки
		cin >> output_file_name;						//имя файла в который пишем результат
		cin >> epsilon;					//точность
		cin >> lx >> ly >> lz;	//размеры области
		cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back; // граничные условия
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

//---------------------------------------------------------------------------------// Вычисление индексов(переход к трехмерной индексации)
	
	kb = id / (nb_X * nb_Y);
	jb = id % (nb_X * nb_Y) / nb_X;
	ib = id % (nb_X * nb_Y) % nb_X;
	
	hx = lx / (nX * nb_X);
	hy = ly / (nY * nb_Y);
	hz = lz / (nZ * nb_Z);
	diff = epsilon + 1;

//---------------------------------------------------------------------------------// Выделение памяти (n+2 из-за дополнительных ячеек)

	data = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));
	next = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2) * (nZ + 2));
	buff = (double*)malloc(sizeof(double) * max(nX, nY) * max(nY, nZ));

//---------------------------------------------------------------------------------// Определение рзмер буфера

	MPI_Pack_size(max(nX, nY) * max(nY, nZ), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

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

	while (true) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nb_X) {
			for (j = 0; j < nY; j++) {
				for (k = 0; k < nZ; k++) {
					buff[j * nZ + k] = data[_i(nX - 1, j, k)];
				}
			}
			MPI_Bsend(buff, nY * nZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nb_Y) {
			for (i = 0; i < nX; i++) {
				for (k = 0; k < nZ; k++) {
					buff[i * nZ + k] = data[_i(i, nY - 1, k)];
				}
			}
			MPI_Bsend(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < nb_Z) {
			for (i = 0; i < nX; i++) {
				for (j = 0; j < nY; j++) {
					buff[i * nY + j] = data[_i(i, j, nZ - 1)];
				}
			}
			MPI_Bsend(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			for (j = 0; j < nY; j++) {
				for (k = 0; k < nZ; k++) {
					buff[j * nZ + k] = data[_i(0, j, k)];
				}
			}
			MPI_Bsend(buff, nY * nZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			for (i = 0; i < nX; i++) {
				for (k = 0; k < nZ; k++) {
					buff[i * nZ + k] = data[_i(i, 0, k)];
				}
			}
			MPI_Bsend(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			for (i = 0; i < nX; i++) {
				for (j = 0; j < nY; j++) {
					buff[i * nY + j] = data[_i(i, j, 0)];
				}
			}
			MPI_Bsend(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

//---------------------------------------------------------------------------------// Прием данных

		if (ib > 0) {
			MPI_Recv(buff, nY * nZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
			for (j = 0; j < nY; j++) {
				for (k = 0; k < nZ; k++) {
					data[_i(-1, j, k)] = buff[j * nZ + k];
				}
			}
		}

		if (jb > 0) {
			MPI_Recv(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
			for (i = 0; i < nX; i++) {
				for (k = 0; k < nZ; k++) {
					data[_i(i, -1, k)] = buff[i * nZ + k];
				}
			}
		}

		if (kb > 0) {
			MPI_Recv(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < nX; i++) {
				for (j = 0; j < nY; j++) {
					data[_i(i, j, -1)] = buff[i * nY + j];
				}
			}
		}

		if (ib + 1 < nb_X) {
			MPI_Recv(buff, nY * nZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
			for (j = 0; j < nY; j++) {
				for (k = 0; k < nZ; k++) {
					data[_i(nX, j, k)] = buff[j * nZ + k];
				}
			}
		}

		if (jb + 1 < nb_Y) {
			MPI_Recv(buff, nX * nZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
			for (i = 0; i < nX; i++) {
				for (k = 0; k < nZ; k++) {
					data[_i(i, nY, k)] = buff[i * nZ + k];
				}
			}
		}

		if (kb + 1 < nb_Z) {
			MPI_Recv(buff, nX * nY, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
			for (i = 0; i < nX; i++) {
				for (j = 0; j < nY; j++) {
					data[_i(i, j, nZ)] = buff[i * nY + j];
				}
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
//---------------------------------------------------------------------------------// Обмен и Allreduce
		diff = 0.0;
		for (i = 0; i < nX; i++) {
			for (j = 0; j < nY; j++) {
				for (k = 0; k < nZ; k++) {
					next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
						(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
						(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

					diff = fmax(diff, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
				}
			}
		}



		temp = next;
		next = data;
		data = temp;
		MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if(diff < epsilon){
			break;
		}
	}

//---------------------------------------------------------------------------------// Вычислительный цикл


//---------------------------------------------------------------------------------// Вывод днных

	if (id != 0) {
		for (k = 0; k < nZ; k++) {
			for (j = 0; j < nY; j++) {
				for (i = 0; i < nX; i++) {
					buff[i] = data[_i(i, j, k)];
				}
				MPI_Send(buff, nX, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	} else {
		fstream out_stream(output_file_name, ios::out);
        out_stream << scientific << setprecision(6);
		for (kb = 0; kb < nb_Z; kb++) {
			for (k = 0; k < nZ; k++) {
				for (jb = 0; jb < nb_Y; jb++) {
					for (j = 0; j < nY; j++) {
						for (ib = 0; ib < nb_X; ib++) {
							if (_ib(ib, jb, kb) == 0) {
								for (i = 0; i < nX; i++) {
									buff[i] = data[_i(i, j, k)];
								}
							} else {
								MPI_Recv(buff, nX, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							}
							for (i = 0; i < nX; i++) {
								out_stream << buff[i] << ' ';
							}
						}
					}
				}
			}
		}
		out_stream.close();
	}
	MPI_Finalize();

	return 0;	
}