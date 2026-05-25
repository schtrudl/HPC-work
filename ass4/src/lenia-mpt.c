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

int myid, procs;
int tiles_one_dim = 0;
int tile_size = 0;
int my_coords[2];
int north_west, north, north_east, west, east, south_west, south, south_east;
MPI_Comm cart_comm;

fx* my_world_top_halo;
fx* my_world;

static inline void exchange_halo(MPI_Datatype row_type, MPI_Datatype col_type, MPI_Datatype corner_type, fx* const grid, fx* const core) {
    const int stride = tile_size + 2 * HALO;

    // Send to north, receive from south
    MPI_Sendrecv(core, 1, row_type, north, 0, grid + (HALO + tile_size) * stride + HALO, 1, row_type, south, 0, cart_comm, MPI_STATUS_IGNORE);
    // Send to south, receive from north
    MPI_Sendrecv(core + (tile_size - HALO) * stride, 1, row_type, south, 1, grid + HALO, 1, row_type, north, 1, cart_comm, MPI_STATUS_IGNORE);
    // Send to west, receive from east
    MPI_Sendrecv(core, 1, col_type, west, 2, grid + HALO * stride + (HALO + tile_size), 1, col_type, east, 2, cart_comm, MPI_STATUS_IGNORE);
    // Send to east, receive from west
    MPI_Sendrecv(core + (tile_size - HALO), 1, col_type, east, 3, grid + HALO * stride, 1, col_type, west, 3, cart_comm, MPI_STATUS_IGNORE);
    // Send to north-west corner
    MPI_Sendrecv(core, 1, corner_type, north_west, 4, grid + (HALO + tile_size) * stride + (HALO + tile_size), 1, corner_type, south_east, 4, cart_comm, MPI_STATUS_IGNORE);
    // Send to north-east corner
    MPI_Sendrecv(core + (tile_size - HALO), 1, corner_type, north_east, 5, grid + (HALO + tile_size) * stride, 1, corner_type, south_west, 5, cart_comm, MPI_STATUS_IGNORE);
    // Send to south-west corner
    MPI_Sendrecv(core + (tile_size - HALO) * stride, 1, corner_type, south_west, 6, grid + (HALO + tile_size), 1, corner_type, north_east, 6, cart_comm, MPI_STATUS_IGNORE);
    // Send to south-east corner
    MPI_Sendrecv(core + (tile_size - HALO) * stride + (tile_size - HALO), 1, corner_type, south_east, 7, grid, 1, corner_type, north_west, 7, cart_comm, MPI_STATUS_IGNORE);
}

int main(int argc, char* argv[]) {
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &procs);  // number of processes involved in communication
    MPI_Get_processor_name(node_name, &name_len);  // compute node name
    const bool master = (myid == 0);
    for (int i = 1; i <= procs; i++) {
        if (i * i == procs) {
            tiles_one_dim = i;
            break;
        }
    }
    if (tiles_one_dim <= 0) {
        if (master) {
            fprintf(stderr, "Error: %d processes is not a perfect square. Use 1, 4, 16, ...\n", procs);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 1;
    }
    if (SIZE % tiles_one_dim != 0) {
        if (master) {
            fprintf(stderr, "Error: SIZE (%d) must be divisible by sqrt(procs) (%d). Use a different SIZE or number of processes.\n", SIZE, tiles_one_dim);
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
        return 1;
    }

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

    tile_size = SIZE / tiles_one_dim;
    if (tile_size < HALO) {
        if (master) {
            fprintf(stderr, "Error: Each process must have at least %d rows and columns (current: %d). Use fewer processes or smaller SIZE.\n", HALO + 1, tile_size);
        }
        MPI_Abort(MPI_COMM_WORLD, 3);
        return 1;
    }

    int dims[2] = {tiles_one_dim, tiles_one_dim};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
    MPI_Comm_rank(cart_comm, &myid);
    MPI_Cart_coords(cart_comm, myid, 2, my_coords);
    /*
    NW N NE
    W  .  E
    SW S SE
    */
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    int diag_coords[2];
    diag_coords[0] = (my_coords[0] - 1 + tiles_one_dim) % tiles_one_dim;
    diag_coords[1] = (my_coords[1] - 1 + tiles_one_dim) % tiles_one_dim;
    MPI_Cart_rank(cart_comm, diag_coords, &north_west);
    diag_coords[0] = (my_coords[0] - 1 + tiles_one_dim) % tiles_one_dim;
    diag_coords[1] = (my_coords[1] + 1) % tiles_one_dim;
    MPI_Cart_rank(cart_comm, diag_coords, &north_east);
    diag_coords[0] = (my_coords[0] + 1) % tiles_one_dim;
    diag_coords[1] = (my_coords[1] - 1 + tiles_one_dim) % tiles_one_dim;
    MPI_Cart_rank(cart_comm, diag_coords, &south_west);
    diag_coords[0] = (my_coords[0] + 1) % tiles_one_dim;
    diag_coords[1] = (my_coords[1] + 1) % tiles_one_dim;
    MPI_Cart_rank(cart_comm, diag_coords, &south_east);

    // Allocate memory
    fx* const k = (fx*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(fx));
    fx* const world = (fx*)calloc(SIZE * SIZE, sizeof(fx));
    fx* const tmp = (fx*)calloc(tile_size * tile_size, sizeof(fx));

    /*
    HALO   HALO   HALO
    HALO MY_WORLD HALO
    HALO   HALO   HALO
    */

    int my_world_stride = tile_size + 2 * HALO;
    my_world_top_halo = (fx*)calloc(my_world_stride * my_world_stride, sizeof(fx));
    my_world = &my_world_top_halo[HALO * my_world_stride + HALO];

    // Generate convolution kernel
    generate_kernel(k, KERNEL_SIZE);
    const fx* const w = k;

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        place_orbium(world, SIZE, SIZE, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // --- 1. Definiranje podatkovnega tipa za pošiljanje ene ploščice (Tile) iz matrike 'world' ---
    MPI_Datatype send_tile_type, resized_send_tile_type;
    int world_sizes[2] = {SIZE, SIZE};
    int tile_sizes[2] = {tile_size, tile_size};
    int send_starts[2] = {0, 0};

    MPI_Type_create_subarray(2, world_sizes, tile_sizes, send_starts, MPI_ORDER_C, MPI_FLOAT, &send_tile_type);  // Opomba: Če je fx 'double', zamenjajte z MPI_DOUBLE

    // Spremenimo obseg tipa na velikost enega elementa (fx), da omogočimo poljubne odmike (displacements) v bajtih
    MPI_Type_create_resized(send_tile_type, 0, sizeof(fx), &resized_send_tile_type);
    MPI_Type_commit(&resized_send_tile_type);

    // --- 2. Definiranje podatkovnega tipa za sprejem ploščice v lokalni 'my_world_top_halo' ---
    // Ker ima lokalna matrika okoli sebe HALO robove, moramo podatke vpisati točno v sredino (kamor kaže my_world)
    MPI_Datatype recv_tile_type;
    int local_sizes[2] = {tile_size + 2 * HALO, tile_size + 2 * HALO};
    int recv_starts[2] = {HALO, HALO};  // Vpisovati začnemo na indeksu HALO, HALO

    MPI_Type_create_subarray(2, local_sizes, tile_sizes, recv_starts, MPI_ORDER_C, MPI_FLOAT, &recv_tile_type);
    MPI_Type_commit(&recv_tile_type);

    int* sendcounts = NULL;
    int* displs = NULL;

    if (master) {
        sendcounts = (int*)calloc(procs, sizeof(int));
        displs = (int*)calloc(procs, sizeof(int));

        for (int r = 0; r < procs; r++) {
            sendcounts[r] = 1;  // Vsak proces prejme natanko 1 ploščico

            // Poiščemo kartezijske koordinate ciljnega procesa 'r'
            int target_coords[2];
            MPI_Cart_coords(cart_comm, r, 2, target_coords);

            // Izračunamo začetni indeks (vrstica, stolpec) ploščice v globalni matriki 'world'
            int row_start = target_coords[0] * tile_size;
            int col_start = target_coords[1] * tile_size;

            // Odmik (displacement) v enotah elementov tipa fx
            displs[r] = row_start * SIZE + col_start;
        }
    }

    MPI_Scatterv(world, sendcounts, displs, resized_send_tile_type, my_world_top_halo, 1, recv_tile_type, 0, cart_comm);

    MPI_Datatype row_type;
    MPI_Datatype col_type;
    MPI_Datatype corner_type;
    MPI_Type_vector(HALO, tile_size, (tile_size + 2 * HALO), MPI_FLOAT, &row_type);
    MPI_Type_vector(tile_size, HALO, (tile_size + 2 * HALO), MPI_FLOAT, &col_type);
    MPI_Type_vector(HALO, HALO, tile_size + 2 * HALO, MPI_FLOAT, &corner_type);
    MPI_Type_commit(&row_type);
    MPI_Type_commit(&col_type);
    MPI_Type_commit(&corner_type);

    const f64 start = MPI_Wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        exchange_halo(row_type, col_type, corner_type, my_world_top_halo, my_world);

        // Convolution
        for (int y = 0; y < tile_size; y++) {
            for (int x = 0; x < tile_size; x++) {
                fx sum = 0;
                for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        const int r = y + kri;
                        const int c = x + kcj;
                        sum += w(ki, kj) * my_world_top_halo[r * my_world_stride + c];
                    }
                }
                tmp[y * tile_size + x] = sum;
            }
        }

        // Evolution
        for (int y = 0; y < tile_size; y++) {
            for (int x = 0; x < tile_size; x++) {
                my_world[y * my_world_stride + x] += DT * growth_lenia(tmp[y * tile_size + x]);
                my_world[y * my_world_stride + x] = fminf(1, fmaxf(0, my_world[y * my_world_stride + x]));  // Clip between 0 and 1
            }
        }
#ifdef GENERATE_GIF
        MPI_Gatherv(my_world_top_halo, 1, recv_tile_type, world, sendcounts, displs, resized_send_tile_type, 0, cart_comm);
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