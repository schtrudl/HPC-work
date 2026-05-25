#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "orbium.h"
#include "gifenc.h"
#include <mpi.h>
#include "stdbool.h"
#include "const_kernel.h"

#ifndef SIZE
    #define SIZE 64
#endif

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

struct orbium_coo {
    int row;
    int col;
    int angle;
};

// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * KERNEL_SIZE + (c)])

// Function to calculate Gaussian
static inline fx gauss(const fx x, const fx mu, const fx sigma) {
    const fx z = (x - mu) / sigma;
    return expf(-0.5f * z * z);
}

// Function for growth criteria
static inline fx growth_lenia(const fx u) {
    const fx mu = 0.15f;
    const fx sigma = 0.015f;
    return -1 + 2 * gauss(u, mu, sigma);  // Baseline -1, peak +1
}

// Polynomial approximation of growth_lenia (max error: 1.26e-05)
static inline f32 growth_lenia_poly(const f32 u) {
    const f32 mu = 0.15f, sigma = 0.015f;
    if (u < 0.09f || u > 0.21f) return -1.0f;
    const f32 t = (u - mu) / sigma, t2 = t * t;
    return ((((((((((1.5164566482e-11f) * t2 + (-1.5024997416e-09f)) * t2 + (6.7570921873e-08f)) * t2 + (-1.8452074685e-06f)) * t2 + (3.4634562942e-05f)) * t2 + (-4.7938982646e-04f)) * t2 + (5.0822995568e-03f)) * t2 + (-4.1438623715e-02f)) * t2 + (2.4978575127e-01f)) * t2 + (-9.9992089585e-01f)) * t2 + (9.9999515209e-01f);
}

// Function to generate convolution kernel
fx* generate_kernel(fx* const K, const unsigned int size) {
    // Construct ring convolution filter
    const fx mu = 0.5f;
    const fx sigma = 0.15f;
    const int r = size / 2;
    fx sum = 0;

    for (int y = -r; y < r; y++) {
        for (int x = -r; x < r; x++) {
            const fx distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
            K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
            if (distance > 1) {
                K[(y + r) * size + x + r] = 0;  // Cut at d=1
            }
            sum += K[(y + r) * size + x + r];
        }
    }
    // Normalize
    for (unsigned int y = 0; y < size; y++) {
        for (unsigned int x = 0; x < size; x++) {
            K[y * size + x] /= sum;
        }
    }
    return K;
}

struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, SIZE / 3, 0}, {SIZE / 3, 0, 180}};

#define HALO KERNEL_SIZE / 2  // Number of halo rows for border exchange
#define TAG_TOP_BOTTOM 10
#define TAG_BOTTOM_TOP 11
#define TAG_LEFT_RIGHT 12
#define TAG_RIGHT_LEFT 13
#define TAG_TL_BR 14
#define TAG_BR_TL 15
#define TAG_TR_BL 16
#define TAG_BL_TR 17

int myid, procs;
int tile_size;
int proc_rows, proc_cols;
int my_coords[2];
int north, south, west, east;
int northwest, northeast, southwest, southeast;
int local_rows, local_cols;
int start_row, start_col;
int pitch;
MPI_Comm cart_comm;

fx* my_world_top_halo;
fx* my_world;

static inline int local_extent(const int global_size, const int parts, const int coord) {
    const int base = global_size / parts;
    const int rem = global_size % parts;
    return base + (coord < rem);
}

static inline int local_start(const int global_size, const int parts, const int coord) {
    const int base = global_size / parts;
    const int rem = global_size % parts;
    return coord * base + ((coord < rem) ? coord : rem);
}

static inline bool choose_tiling(const int nprocs, int* rows, int* cols) {
    const int root = (int)(sqrt((double)nprocs) + 0.5);
    if (root * root != nprocs) {
        return false;
    }
    *rows = root;
    *cols = root;
    return true;
}

static inline void exchange_halo(MPI_Datatype row_type, MPI_Datatype col_type, MPI_Datatype corner_type, fx* const grid, fx* const core) {
    // Exchange top/bottom strips.
    MPI_Sendrecv(core, 1, row_type, north, TAG_TOP_BOTTOM, grid + (HALO + local_rows) * pitch + HALO, 1, row_type, south, TAG_TOP_BOTTOM, cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(core + (local_rows - HALO) * pitch, 1, row_type, south, TAG_BOTTOM_TOP, grid + HALO, 1, row_type, north, TAG_BOTTOM_TOP, cart_comm, MPI_STATUS_IGNORE);

    // Exchange left/right strips.
    MPI_Sendrecv(core, 1, col_type, west, TAG_LEFT_RIGHT, grid + HALO * pitch + HALO + local_cols, 1, col_type, east, TAG_LEFT_RIGHT, cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(core + (local_cols - HALO), 1, col_type, east, TAG_RIGHT_LEFT, grid + HALO * pitch, 1, col_type, west, TAG_RIGHT_LEFT, cart_comm, MPI_STATUS_IGNORE);

    // Exchange corner blocks.
    MPI_Sendrecv(core, 1, corner_type, northwest, TAG_TL_BR, grid + (HALO + local_rows) * pitch + HALO + local_cols, 1, corner_type, southeast, TAG_TL_BR, cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(core + (local_rows - HALO) * pitch + (local_cols - HALO), 1, corner_type, southeast, TAG_BR_TL, grid, 1, corner_type, northwest, TAG_BR_TL, cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(core + (local_cols - HALO), 1, corner_type, northeast, TAG_TR_BL, grid + (HALO + local_rows) * pitch, 1, corner_type, southwest, TAG_TR_BL, cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(core + (local_rows - HALO) * pitch, 1, corner_type, southwest, TAG_BL_TR, grid + HALO + local_cols, 1, corner_type, northeast, TAG_BL_TR, cart_comm, MPI_STATUS_IGNORE);
}

static inline void scatter_world(const fx* const world, fx* const local, const bool master) {
    fx* recv_block = (fx*)malloc((size_t)local_rows * (size_t)local_cols * sizeof(fx));
    if (master) {
        for (int rank = 0; rank < procs; rank++) {
            int coords[2];
            MPI_Cart_coords(cart_comm, rank, 2, coords);
            const int rr = local_extent(SIZE, proc_rows, coords[0]);
            const int cc = local_extent(SIZE, proc_cols, coords[1]);
            const int sr = local_start(SIZE, proc_rows, coords[0]);
            const int sc = local_start(SIZE, proc_cols, coords[1]);
            fx* block = (fx*)malloc((size_t)rr * (size_t)cc * sizeof(fx));
            for (int r = 0; r < rr; r++) {
                memcpy(block + r * cc, world + (size_t)(sr + r) * SIZE + sc, (size_t)cc * sizeof(fx));
            }
            if (rank == 0) {
                for (int r = 0; r < rr; r++) {
                    memcpy(local + (size_t)r * pitch, block + (size_t)r * cc, (size_t)cc * sizeof(fx));
                }
            } else {
                MPI_Send(block, rr * cc, MPI_FLOAT, rank, 1000, cart_comm);
            }
            free(block);
        }
    } else {
        MPI_Recv(recv_block, local_rows * local_cols, MPI_FLOAT, 0, 1000, cart_comm, MPI_STATUS_IGNORE);
        for (int r = 0; r < local_rows; r++) {
            memcpy(local + (size_t)r * pitch, recv_block + (size_t)r * local_cols, (size_t)local_cols * sizeof(fx));
        }
    }
    free(recv_block);
}

static inline void gather_world(fx* const world, const fx* const local, const bool master) {
    fx* send_block = (fx*)malloc((size_t)local_rows * (size_t)local_cols * sizeof(fx));
    if (master) {
        for (int rank = 0; rank < procs; rank++) {
            int coords[2];
            MPI_Cart_coords(cart_comm, rank, 2, coords);
            const int rr = local_extent(SIZE, proc_rows, coords[0]);
            const int cc = local_extent(SIZE, proc_cols, coords[1]);
            const int sr = local_start(SIZE, proc_rows, coords[0]);
            const int sc = local_start(SIZE, proc_cols, coords[1]);

            fx* block = (fx*)malloc((size_t)rr * (size_t)cc * sizeof(fx));
            if (rank == 0) {
                for (int r = 0; r < rr; r++) {
                    memcpy(block + (size_t)r * cc, local + (size_t)r * pitch, (size_t)cc * sizeof(fx));
                }
            } else {
                MPI_Recv(block, rr * cc, MPI_FLOAT, rank, 1001, cart_comm, MPI_STATUS_IGNORE);
            }
            for (int r = 0; r < rr; r++) {
                memcpy(world + (size_t)(sr + r) * SIZE + sc, block + r * cc, (size_t)cc * sizeof(fx));
            }
            free(block);
        }
    } else {
        for (int r = 0; r < local_rows; r++) {
            memcpy(send_block + (size_t)r * local_cols, local + (size_t)r * pitch, (size_t)local_cols * sizeof(fx));
        }
        MPI_Send(send_block, local_rows * local_cols, MPI_FLOAT, 0, 1001, cart_comm);
    }
    free(send_block);
}

static inline void gather_packed_world(fx* const world, const fx* const local, const bool master) {
    if (master) {
        for (int rank = 0; rank < procs; rank++) {
            int coords[2];
            MPI_Cart_coords(cart_comm, rank, 2, coords);
            const int rr = local_extent(SIZE, proc_rows, coords[0]);
            const int cc = local_extent(SIZE, proc_cols, coords[1]);
            const int sr = local_start(SIZE, proc_rows, coords[0]);
            const int sc = local_start(SIZE, proc_cols, coords[1]);

            fx* block = (fx*)malloc((size_t)rr * (size_t)cc * sizeof(fx));
            if (rank == 0) {
                memcpy(block, local, (size_t)rr * (size_t)cc * sizeof(fx));
            } else {
                MPI_Recv(block, rr * cc, MPI_FLOAT, rank, 2001, cart_comm, MPI_STATUS_IGNORE);
            }
            for (int r = 0; r < rr; r++) {
                memcpy(world + (size_t)(sr + r) * SIZE + sc, block + (size_t)r * cc, (size_t)cc * sizeof(fx));
            }
            free(block);
        }
    } else {
        MPI_Send((void*)local, local_rows * local_cols, MPI_FLOAT, 0, 2001, cart_comm);
    }
}

int main(int argc, char* argv[]) {
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &procs);  // number of processes involved in communication
    MPI_Get_processor_name(node_name, &name_len);  // compute node name
    const bool master = (myid == 0);
    if (!choose_tiling(procs, &proc_rows, &proc_cols)) {
        if (master) {
            fprintf(stderr, "Error: %d processes is not a perfect square. Use 1, 4, 9, 16, 25, ...\n", procs);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int dims[2] = {proc_rows, proc_cols};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &myid);
    MPI_Cart_coords(cart_comm, myid, 2, my_coords);

    local_rows = local_extent(SIZE, proc_rows, my_coords[0]);
    local_cols = local_extent(SIZE, proc_cols, my_coords[1]);
    start_row = local_start(SIZE, proc_rows, my_coords[0]);
    start_col = local_start(SIZE, proc_cols, my_coords[1]);
    pitch = local_cols + 2 * HALO;

    if (local_rows < HALO || local_cols < HALO) {
        if (master) {
            fprintf(stderr, "Error: Each process must have at least %d rows and columns (current: %d rows, %d cols). Use fewer processes or smaller SIZE.\n", HALO + 1, local_rows, local_cols);
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    int diag_coords[2];
    diag_coords[0] = (my_coords[0] - 1 + proc_rows) % proc_rows;
    diag_coords[1] = (my_coords[1] - 1 + proc_cols) % proc_cols;
    MPI_Cart_rank(cart_comm, diag_coords, &northwest);
    diag_coords[0] = (my_coords[0] - 1 + proc_rows) % proc_rows;
    diag_coords[1] = (my_coords[1] + 1) % proc_cols;
    MPI_Cart_rank(cart_comm, diag_coords, &northeast);
    diag_coords[0] = (my_coords[0] + 1) % proc_rows;
    diag_coords[1] = (my_coords[1] - 1 + proc_cols) % proc_cols;
    MPI_Cart_rank(cart_comm, diag_coords, &southwest);
    diag_coords[0] = (my_coords[0] + 1) % proc_rows;
    diag_coords[1] = (my_coords[1] + 1) % proc_cols;
    MPI_Cart_rank(cart_comm, diag_coords, &southeast);
    //printf("Hello from process %d of %d in node %s\n", myid, procs, node_name);

#ifdef GENERATE_GIF
    ge_GIF* gif = NULL;
    if (master) {
        gif = ge_new_gif(
            "lenia.gif", /* file name */
            SIZE,
            SIZE, /* canvas size */
            (uint8_t*)inferno_pallete, /*pallete*/
            8, /* palette depth == log2(# of colors) */
            -1, /* no transparency */
            0 /* infinite loop */
        );
    }
#endif

    // Allocate memory
    fx* const k = (fx*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(fx));
    fx* const world = (fx*)calloc(SIZE * SIZE, sizeof(fx));
    fx* const tmp = (fx*)calloc((size_t)local_rows * (size_t)local_cols, sizeof(fx));

    const size_t local_total = (size_t)(local_rows + 2 * HALO) * (size_t)(local_cols + 2 * HALO);
    my_world_top_halo = (fx*)calloc(local_total, sizeof(fx));
    my_world = my_world_top_halo + (size_t)HALO * pitch + HALO;

    // Generate convolution kernel
    generate_kernel(k, KERNEL_SIZE);
    const fx* const w = k;

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        place_orbium(world, SIZE, SIZE, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    scatter_world(world, my_world, master);

    MPI_Datatype row_type;
    MPI_Datatype col_type;
    MPI_Datatype corner_type;
    MPI_Type_vector(HALO, local_cols, pitch, MPI_FLOAT, &row_type);
    MPI_Type_vector(local_rows, HALO, pitch, MPI_FLOAT, &col_type);
    MPI_Type_vector(HALO, HALO, pitch, MPI_FLOAT, &corner_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);
    MPI_Type_commit(&corner_type);

    const f64 start = MPI_Wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        exchange_halo(row_type, col_type, corner_type, my_world_top_halo, my_world);

        // Convolution
        for (int y = 0; y < local_rows; y++) {
            for (int x = 0; x < local_cols; x++) {
                fx sum = 0;
                for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        const int r = y + kri;
                        const int c = x + kcj;
                        sum += w(ki, kj) * my_world_top_halo[r * pitch + c];
                    }
                }
                tmp[y * local_cols + x] = sum;
            }
        }

        // Evolution
        for (int y = 0; y < local_rows; y++) {
            for (int x = 0; x < local_cols; x++) {
                my_world[y * pitch + x] += DT * growth_lenia(tmp[y * local_cols + x]);
                my_world[y * pitch + x] = fminf(1, fmaxf(0, my_world[y * pitch + x]));  // Clip between 0 and 1
            }
        }
#ifdef GENERATE_GIF
        gather_world(world, my_world, master);
        if (master) {
            for (unsigned int i = 0; i < SIZE * SIZE; i++) {
                gif->frame[i] = world[i] * 255;
            }
            ge_add_frame(gif, 5);
        }
#endif
    }
    const f64 stop = MPI_Wtime();
    if (master) {
        printf("Time(full): %f s\n", stop - start);
#ifdef GENERATE_GIF
        ge_close_gif(gif);
#endif
    }
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    MPI_Type_free(&corner_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    free(my_world_top_halo);
    free(k);
    free(tmp);
    free(world);
    return 0;
}