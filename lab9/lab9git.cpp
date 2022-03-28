#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <omp.h>


#define _i(i, j, k) (((k) + 1) * (nx + 2) * (ny + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * nbx * nby + (j) * nbx + (i))


int main(int argc, char* argv[]) {
	int ib, jb, kb, nbx, nby, nbz, nx, ny, nz;
	int id, numproc;
	double lx, ly, lz, hx, hy, hz, bc_down, bc_up, bc_left, bc_right, bc_front, bc_back, u0;
	double eps, cur_eps;
	double *data, *next, *temp;
	char fname[100];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if (id == 0) {
		scanf("%d %d %d", &nbx, &nby, &nbz);
		scanf("%d %d %d", &nx, &ny, &nz);
		scanf("%s", fname);
		scanf("%lf", &eps);
		scanf("%lf %lf %lf", &lx, &ly, &lz);
		scanf("%lf %lf %lf %lf %lf %lf", &bc_down, &bc_up, &bc_left, &bc_right, &bc_front, &bc_back);
		scanf("%lf", &u0);
	}

	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbz, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(fname, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

	kb = id / (nbx * nby);
	jb = id % (nbx * nby) / nbx;
	ib = id % (nbx * nby) % nbx;

	hx = lx / (nx * nbx);
	hy = ly / (ny * nby);
	hz = lz / (nz * nbz);

	int sizes[] = {nz + 2, ny + 2, nx + 2};
	MPI_Datatype left_bnd_send;
	int subsizes_lr[] = {nz, ny, 1};
	int starts_lfd[] = {1, 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &left_bnd_send);
	MPI_Type_commit(&left_bnd_send);

	MPI_Datatype front_bnd_send;
	int subsizes_fb[] = {nz, 1, nx};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &front_bnd_send);
	MPI_Type_commit(&front_bnd_send);

	MPI_Datatype down_bnd_send;
	int subsizes_du[] = {1, ny, nx};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_lfd, MPI_ORDER_C, MPI_DOUBLE, &down_bnd_send);
	MPI_Type_commit(&down_bnd_send);

	MPI_Datatype right_bnd_send;
	int starts_nx[] = {1, 1, nx};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_nx, MPI_ORDER_C, MPI_DOUBLE, &right_bnd_send);
	MPI_Type_commit(&right_bnd_send);

	MPI_Datatype back_bnd_send;
	int starts_ny[] = {1, ny, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_ny, MPI_ORDER_C, MPI_DOUBLE, &back_bnd_send);
	MPI_Type_commit(&back_bnd_send);

	MPI_Datatype up_bnd_send;
	int starts_nz[] = {nz, 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_nz, MPI_ORDER_C, MPI_DOUBLE, &up_bnd_send);
	MPI_Type_commit(&up_bnd_send);

/*-----------------------------------------------------------------------------------*/
	MPI_Datatype left_bnd_recv;
	int starts_x0[] = {1, 1, 0};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_x0, MPI_ORDER_C, MPI_DOUBLE, &left_bnd_recv);
	MPI_Type_commit(&left_bnd_recv);

	MPI_Datatype front_bnd_recv;
	int starts_y0[] = {1, 0, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_y0, MPI_ORDER_C, MPI_DOUBLE, &front_bnd_recv);
	MPI_Type_commit(&front_bnd_recv);

	MPI_Datatype down_bnd_recv;
	int starts_z0[] = {0, 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_z0, MPI_ORDER_C, MPI_DOUBLE, &down_bnd_recv);
	MPI_Type_commit(&down_bnd_recv);

	MPI_Datatype right_bnd_recv;
	int starts_r[] = {1, 1, nx + 1};
	MPI_Type_create_subarray(3, sizes, subsizes_lr, starts_r, MPI_ORDER_C, MPI_DOUBLE, &right_bnd_recv);
	MPI_Type_commit(&right_bnd_recv);

	MPI_Datatype back_bnd_recv;
	int starts_y[] = {1, ny + 1, 1};
	MPI_Type_create_subarray(3, sizes, subsizes_fb, starts_y, MPI_ORDER_C, MPI_DOUBLE, &back_bnd_recv);
	MPI_Type_commit(&back_bnd_recv);

	MPI_Datatype up_bnd_recv;
	int starts_z[] = {nz + 1, 1,  1};
	MPI_Type_create_subarray(3, sizes, subsizes_du, starts_z, MPI_ORDER_C, MPI_DOUBLE, &up_bnd_recv);
	MPI_Type_commit(&up_bnd_recv);

	
	data = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
	next = (double*)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));

	int buf_side1 = std::max(nx, ny);
	int buf_side2 = std::max(ny, nz);

	int buffer_size;
	MPI_Pack_size(buf_side1 * buf_side2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 6 * (buffer_size + MPI_BSEND_OVERHEAD);
	double* buffer = (double*)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	for (int i = 0; i < nx; i++) {					// Инициализация блока
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				data[_i(i, j, k)] = u0;
			}
		}
	}

	for (int j = 0; j < ny; j++) {
		for (int k = 0; k < nz; k++) {
			data[_i(-1, j, k)] = bc_left;
			next[_i(-1, j, k)] = bc_left;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int k = 0; k < nz; k++) {
			data[_i(i, -1, k)] = bc_front;
			next[_i(i, -1, k)] = bc_front;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, -1)] = bc_down;
			next[_i(i, j, -1)] = bc_down;
		}
	}

	for (int j = 0; j < ny; j++) {
		for (int k = 0; k < nz; k++) {
			data[_i(nx, j, k)] = bc_right;
			next[_i(nx, j, k)] = bc_right;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int k = 0; k < nz; k++) {
			data[_i(i, ny, k)] = bc_back;
			next[_i(i, ny, k)] = bc_back;
		}
	}

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			data[_i(i, j, nz)] = bc_up;
			next[_i(i, j, nz)] = bc_up;
		}
	}

	cur_eps = eps + 1;
	while (cur_eps >= eps) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {
			MPI_Bsend(data, 1, right_bnd_send, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nby) {
			MPI_Bsend(data, 1, back_bnd_send, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb + 1 < nbz) {
			MPI_Bsend(data, 1, up_bnd_send, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Bsend(data, 1, left_bnd_send, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			MPI_Bsend(data, 1, front_bnd_send, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			MPI_Bsend(data, 1, down_bnd_send, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
		}

		/*-----------------------------------------------------------------------------------*/
		if (ib > 0) {
			MPI_Recv(data, 1, left_bnd_recv, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
		}

		if (jb > 0) {
			MPI_Recv(data, 1, front_bnd_recv, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
		}

		if (kb > 0) {
			MPI_Recv(data, 1, down_bnd_recv, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
		}

		if (ib + 1 < nbx) {
			MPI_Recv(data, 1, right_bnd_recv, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
		}

		if (jb + 1 < nby) {
			MPI_Recv(data, 1, back_bnd_recv, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
		}

		if (kb + 1 < nbz) {
			MPI_Recv(data, 1, up_bnd_recv, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		cur_eps = 0.0;
		#pragma omp parallel shared(data, next) reduction(max:cur_eps)
		{
			int thread_id = omp_get_thread_num();
			int offset = omp_get_num_threads();

			for (int t = thread_id; t < nx * ny * nz; t += offset) {
				int i = t % nx;
				int j = t % (nx * ny) / nx;
				int k = t / (nx * ny);
				next[_i(i, j, k)] = 0.5 * ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
					(data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
					(data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
					(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));

				cur_eps = fmax(cur_eps, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
			}
		}

		MPI_Allreduce(MPI_IN_PLACE, &cur_eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		temp = next;
		next = data;
		data = temp;
	}

	int n_size = 14;	//знак(-) + мантисса + порядок + '\0' = 1 + 8 + 4 + 1
	char* bf = (char*)malloc(sizeof(char) * nx * ny * nz * n_size);
	memset(bf, ' ', sizeof(char) * nx * ny * nz * n_size);

	for (int k = 0; k < nz; k++) {
		for (int j = 0; j < ny; j++) {
			for (int i = 0; i < nx; i++) {
				sprintf(bf + (k * nx * ny + j * nx + i) * n_size, "%.6e", data[_i(i, j, k)]);
			}
		}
	}

	for (int i = 0; i < nx * ny * nz * n_size; i++) {
		if (bf[i] == '\0') {
			bf[i] = ' ';
		}
	}

	MPI_File fp;
	MPI_Datatype filetype;
	int sizes_gr[] = { nz * nbz, ny * nby, nx * nbx * n_size };
	int subsizes[] = { nz, ny, nx * n_size };
	int starts[] = { id / (nbx * nby) * nz, id % (nbx * nby) / nbx * ny,  id % (nbx * nby) % nbx * nx * n_size };
	MPI_Type_create_subarray(3, sizes_gr, subsizes, starts, MPI_ORDER_C, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

	MPI_File_delete(fname, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, bf, nx * ny * nz * n_size, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);

	MPI_Type_free(&filetype);
	MPI_Type_free(&left_bnd_send);
	MPI_Type_free(&left_bnd_recv);
	MPI_Type_free(&right_bnd_send);
	MPI_Type_free(&right_bnd_recv);
	MPI_Type_free(&front_bnd_send);
	MPI_Type_free(&front_bnd_recv);
	MPI_Type_free(&back_bnd_send);
	MPI_Type_free(&back_bnd_recv);
	MPI_Type_free(&down_bnd_send);
	MPI_Type_free(&down_bnd_recv);
	MPI_Type_free(&up_bnd_send);
	MPI_Type_free(&up_bnd_recv);
	MPI_Finalize();
	free(data);
	free(next);
	free(bf);
	return 0;
}