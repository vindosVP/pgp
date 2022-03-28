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

#define _i(i, j) (((j) + 1) * (nX + 2) + (i) + 1)
#define _ib(i, j) ((j) * nb_X + (i))

using namespace std;

int main(int argc, char* argv[]) {
    int ib, jb, nb_X, nb_Y, nX, nY, numproc, id, buffer_size;
    double lx, ly, hx, hy, bc_down, bc_up, bc_left, bc_right, init_v;
    double epsilon, diff;
    double* data, * temp, * next;
    char output_file_name[100];
    int n_size = 16;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Barrier(MPI_COMM_WORLD);

    if (id == 0) {
        cin >> nb_X >> nb_Y;                                                          //размер блока
        cin >> nX >> nY;                                                                      // размер сетки
        cin >> output_file_name;                                                              // имя файла в который пишем результат
        cin >> epsilon;                                                                       // точность
        cin >> lx >> ly;                                                                  // размеры области
        cin >> bc_left >> bc_right >> bc_down >> bc_up;               // граничные условия
        cin >> init_v;
    }

    MPI_Bcast(&nX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_file_name, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&init_v, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    unsigned long long size = n_size * nX * nY;

    jb = id / nb_X;
    ib = id % nb_X;
    
    int p1 = nX * ib;
    int p2 = nY * jb;
    int p3 = n_size * nX * nb_X;

    hx = lx / (nX * nb_X);
    hy = ly / (nY * nb_Y);
    diff = epsilon + 1;
    MPI_Datatype out_type;
    MPI_Type_create_hvector(nY, n_size * nX, p3, MPI_CHAR, &out_type);

    data = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2));
    next = (double*)malloc(sizeof(double) * (nX + 2) * (nY + 2));

    MPI_Pack_size((max(nX, nY) + 2), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    MPI_Datatype sendrec_lr;
    MPI_Datatype sendrec_ud;
    int part1 = nY + 2;
    int part2 = nX + 2;

    MPI_Type_commit(&out_type);
    MPI_Aint format = p3 * p1 + n_size * p2;

    int i, j;
    #pragma omp parallel for private(i, j) shared(data)
    for (i = 0; i < nX; i++) {
        for (j = 0; j < nY; j++) {
            data[_i(i, j)] = init_v;
        }
    }

    #pragma omp parallel for private(j) shared(data, next)
    for (j = 0; j < nY; j++) {
        data[_i(-1, j)] = bc_left;
        next[_i(-1, j)] = bc_left;
    }

    #pragma omp parallel for private(i) shared(data, next)
    for (i = 0; i < nX; i++) {
        data[_i(i, -1)] = bc_down;
        next[_i(i, -1)] = bc_down;
    }

    #pragma omp parallel for private(j) shared(data, next)
    for (j = 0; j < nY; j++) {
        data[_i(nX, j)] = bc_right;
        next[_i(nX, j)] = bc_right;
    }

    #pragma omp parallel for private(i) shared(data, next)
    for (i = 0; i < nX; i++) {
        data[_i(i, nY)] = bc_up;
        next[_i(i, nY)] = bc_up;
    }
    
    char *text = (char*) malloc(sizeof(char) * size);
    for (int i = 0; i < size; i++) text[i] = ' ';


    MPI_Type_vector(part1, 1, part2, MPI_DOUBLE, &sendrec_lr);
    MPI_Type_vector(1, part1, 0, MPI_DOUBLE, &sendrec_ud);
    MPI_Type_commit(&sendrec_lr);
    MPI_Type_commit(&sendrec_ud);


    while (true) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (ib + 1 < nb_X) MPI_Bsend(data + nX, 1, sendrec_lr, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
        if (jb + 1 < nb_Y) MPI_Bsend(data + (nX + 2) * nY, 1, sendrec_ud, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
        if (ib > 0) MPI_Bsend(data + 1, 1, sendrec_lr, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
        if (jb > 0) MPI_Bsend(data + nX + 2, 1, sendrec_ud, _ib(ib, jb - 1), id, MPI_COMM_WORLD);

        /*---------------------------------------------------------------------------------------------------------*/
        if (ib > 0) MPI_Recv(data, 1, sendrec_lr, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
        if (jb > 0) MPI_Recv(data, 1, sendrec_ud, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
        if (ib + 1 < nb_X) MPI_Recv(data + nX + 1, 1, sendrec_lr, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
        if (jb + 1 < nb_Y) MPI_Recv(data + (nX + 2) * (nY + 1), 1, sendrec_ud, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);


        MPI_Barrier(MPI_COMM_WORLD);
        diff = 0.0;
        #pragma omp parallel for private(i, j) shared(data, next) reduction(max: diff)
        for (int i = 0; i < nX; i++) {
            for (int j = 0; j < nY; j++) {
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

    for (int i = 0; i < nY; i++) 
        for (int j = 0; j < nX; j++) 
            sprintf(text + (i * nX + j) * n_size, "%.7e", data[_i(j, i)]);

    for (size_t i = 0; i < size; i++) 
        if (text[i] == '\0') text[i] = ' ';

    MPI_File output_file;
    MPI_File_delete(output_file_name, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, output_file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_set_view(output_file, format , MPI_CHAR, out_type, "native", MPI_INFO_NULL);
    MPI_File_write_all(output_file, text, size, MPI_CHAR, &status);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}
