#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2048

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int local_rows = SIZE / nprocs;

    // elements each rank receives
    int recvcount = local_rows * SIZE;

    float* global = NULL;

    // root owns full matrix
    if (rank == 0) {
        global = malloc(SIZE * SIZE * sizeof(float));

        for (int i = 0; i < SIZE * SIZE; i++) {
            global[i] = (float)i;
        }
    }

    // local buffer
    float* local = malloc(recvcount * sizeof(float));

    printf("rank=%d recvcount=%d bytes=%zu\n", rank, recvcount, recvcount * sizeof(float));

    int s = MPI_Scatter(
        global,  // send buffer (root only)
        recvcount,  // send count PER RANK
        MPI_FLOAT,  // send datatype

        local,  // receive buffer
        recvcount,  // receive count
        MPI_FLOAT,  // receive datatype

        0,  // root
        MPI_COMM_WORLD);

    printf("%d\n", s);

    printf("rank %d first=%f last=%f\n", rank, local[0], local[recvcount - 1]);

    free(local);

    if (rank == 0) free(global);

    MPI_Finalize();

    return 0;
}