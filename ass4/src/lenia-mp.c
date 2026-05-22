#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#define input(r, c) (my_world[((r) % SIZE) * SIZE + ((c) % SIZE)])

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
#define HALO_SIZE (HALO * SIZE)
#define MY_WORLD_SIZE (my_rows * SIZE)
#define MY_WORLD_TOTAL_SIZE (HALO_SIZE + MY_WORLD_SIZE + HALO_SIZE)

int myid, procs;
fx* my_world_top_halo;
fx* my_world;
fx* my_world_bottom_halo;
unsigned int my_rows;

static inline void exchange_halo() {
    // exchange borders with neighboring processes
    if (my_rows >= HALO) {
        // send top row to previous process
        MPI_Sendrecv(
            // send:
            my_world,
            HALO_SIZE,
            MPI_FLOAT,
            (myid + procs - 1) % procs,
            0,
            // recv:
            my_world_bottom_halo,
            HALO_SIZE,
            MPI_FLOAT,
            (myid + 1) % procs,
            0,
            MPI_COMM_WORLD,
            MPI_STATUSES_IGNORE);
        // send bottom row to next process
        MPI_Sendrecv(
            // send:
            my_world + (my_rows - HALO) * SIZE,
            HALO_SIZE,
            MPI_FLOAT,
            (myid + 1) % procs,
            1,
            // recv:
            my_world_top_halo,
            HALO_SIZE,
            MPI_FLOAT,
            (myid + procs - 1) % procs,
            1,
            MPI_COMM_WORLD,
            MPI_STATUSES_IGNORE);
    } else {  // our world is smaller than HALO so we need data from more neighbors
        int needs = HALO;
        int process_offset = 1;
        int bottom_filled = 0;
        int top_filled = HALO;
        while (needs > 0) {
            int x_rows = (needs < my_rows) ? needs : my_rows;  // min(needs, my_rows)
            // send top x_rows to previous process
            MPI_Sendrecv(
                // send:
                my_world,
                x_rows * SIZE,
                MPI_FLOAT,
                (myid + procs - process_offset) % procs,
                process_offset * 100,
                // recv:
                my_world_bottom_halo + bottom_filled * SIZE,
                x_rows * SIZE,
                MPI_FLOAT,
                (myid + process_offset) % procs,
                process_offset * 100,
                MPI_COMM_WORLD,
                MPI_STATUSES_IGNORE);
            // send bottom row to next process
            MPI_Sendrecv(
                // send:
                my_world + (my_rows - x_rows) * SIZE,
                x_rows * SIZE,
                MPI_FLOAT,
                (myid + process_offset) % procs,
                process_offset * 10000,
                // recv:
                my_world_top_halo + (top_filled - x_rows) * SIZE,
                x_rows * SIZE,
                MPI_FLOAT,
                (myid + procs - process_offset) % procs,
                process_offset * 10000,
                MPI_COMM_WORLD,
                MPI_STATUSES_IGNORE);
            bottom_filled += x_rows;
            top_filled -= x_rows;
            needs -= x_rows;
            process_offset++;
        }
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
    fx* const tmp = (fx*)calloc(SIZE * SIZE, sizeof(fx));

    // Generate convolution kernel
    generate_kernel(k, KERNEL_SIZE);
    const fx* const w = k;

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        place_orbium(world, SIZE, SIZE, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
    my_rows = SIZE / procs;
    const unsigned int my_start = myid * my_rows;
    const unsigned int my_end = my_start + my_rows;

    my_world_top_halo = (fx*)malloc((HALO_SIZE + MY_WORLD_SIZE + HALO_SIZE) * sizeof(fx));
    // fx* my_world_top_halo = &world[(SIZE + my_start - HALO) % SIZE * SIZE];
    my_world = &my_world_top_halo[HALO_SIZE];
    my_world_bottom_halo = &my_world[MY_WORLD_SIZE];
    // Distribute initial world rows to all processes
    // XXX: I guess we do not need to count this
    MPI_Scatter(world, MY_WORLD_SIZE, MPI_FLOAT, my_world, MY_WORLD_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    const f64 start = MPI_Wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        exchange_halo();

        /*
        for (unsigned int y = 0; y < SIZE; y++) {
            for (unsigned int x = 0; x < SIZE; x++) {
                f32 sum = 0;
                for (int k = 0; k < NUM_KERNEL_ENTRIES; k++) {
                    sum += sparse_kernel[k].weight * input((y + sparse_kernel[k].dy + SIZE), (x + sparse_kernel[k].dx + SIZE));
                }
                tmp[y * SIZE + x] = sum;
            }
        }
        */
        // Convolution
        for (unsigned int y = (0 + HALO); y < (my_rows + HALO); y++) {
            for (unsigned int x = 0; x < SIZE; x++) {
                fx sum = 0;
                for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        int r = (y - HALO + kri);
                        int c = (x - HALO + SIZE + kcj);
                        sum += w(ki, kj) * my_world_top_halo[(r * SIZE) + ((c) % SIZE)];
                    }
                }
                tmp[(y - HALO) * SIZE + x] = sum;
            }
        }

        // Evolution
        for (unsigned int y = 0; y < my_rows; y++) {
            for (unsigned int x = 0; x < SIZE; x++) {
                my_world[y * SIZE + x] += DT * growth_lenia(tmp[y * SIZE + x]);
                my_world[y * SIZE + x] = fminf(1, fmaxf(0, my_world[y * SIZE + x]));  // Clip between 0 and 1
            }
        }
#ifdef GENERATE_GIF
        MPI_Gather(my_world, my_rows * SIZE, MPI_FLOAT, world, my_rows * SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    MPI_Finalize();
    free(k);
    free(tmp);
    free(world);
    return 0;
}