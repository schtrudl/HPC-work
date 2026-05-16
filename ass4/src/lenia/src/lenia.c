#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "orbium.h"
#include "gifenc.h"
#include <mpi.h>

struct orbium_coo {
    int row;
    int col;
    int angle;
};

// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma) {
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u) {
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);  // Baseline -1, peak +1
}

// Function to generate convolution kernel
double* generate_kernel(double* K, const unsigned int size) {
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
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
    }
    return K;
}

// Function to perform convolution on input using kernel w
inline double* convolve2d(double* result, const double* input, const double* w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols) {
    if (result != NULL && input != NULL && w != NULL) {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}

// Function to evolve Lenia
double* evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo* orbiums, const unsigned int num_orbiums) {
#ifdef GENERATE_GIF
    ge_GIF* gif = ge_new_gif(
        "lenia.gif", /* file name */
        cols,
        rows, /* canvas size */
        inferno_pallete, /*pallete*/
        8, /* palette depth == log2(# of colors) */
        -1, /* no transparency */
        0 /* infinite loop */
    );
#endif

    // Allocate memory
    double* w = (double*)calloc(kernel_size * kernel_size, sizeof(double));
    double* world = (double*)calloc(rows * cols, sizeof(double));
    double* tmp = (double*)calloc(rows * cols, sizeof(double));

    // Generate convolution kernel
    w = generate_kernel(w, kernel_size);

    // Place orbiums
    for (unsigned int o = 0; o < num_orbiums; o++) {
        world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++) {
        // Convolution
        tmp = convolve2d(tmp, world, w, rows, cols, kernel_size, kernel_size);

        // Evolution
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                world[i * rows + j] += dt * growth_lenia(tmp[i * rows + j]);
                world[i * rows + j] = fmin(1, fmax(0, world[i * rows + j]));  // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * rows + j] = world[i * rows + j] * 255;
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    free(w);
    free(tmp);
    return world;
}

#define N 128
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

int main(int argc, char* argv[]) {
    int myid, procs;
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // process ID
    MPI_Comm_size(MPI_COMM_WORLD, &procs);  // number of processes involved in communication
    MPI_Get_processor_name(node_name, &name_len);  // compute node name
    printf("Hello from process %d of %d in node %s\n", myid, procs, node_name);

    double start = MPI_Wtime();
    double* world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    double stop = MPI_Wtime();
    printf("Execution time: %.3f\n", stop - start);
    free(world);
    MPI_Finalize();
    return 0;
}