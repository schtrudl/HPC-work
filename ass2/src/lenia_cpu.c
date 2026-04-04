#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "orbium.h"
#include "gifenc.h"

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>

// Uncomment to generate gif animation
// #define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

#ifndef N
    #define N 256
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
// Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
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

// Place two orbiums in the world with different angles. (y, x, angle)
// Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

int main() {
#ifdef GENERATE_GIF
    ge_GIF* gif = ge_new_gif(
        "lenia.gif", /* file name */
        N,
        N, /* canvas size */
        inferno_pallete, /*pallete*/
        8, /* palette depth == log2(# of colors) */
        -1, /* no transparency */
        0 /* infinite loop */
    );
#endif

    // Allocate memory
    double* w = (double*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(double));
    double* world = (double*)calloc(N * N, sizeof(double));
    double* tmp = (double*)calloc(N * N, sizeof(double));

    // Generate convolution kernel
    w = generate_kernel(w, KERNEL_SIZE);

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        world = place_orbium(world, N, N, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    double start = omp_get_wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        // Convolution
        tmp = convolve2d(tmp, world, w, N, N, KERNEL_SIZE, KERNEL_SIZE);

        // Evolution
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < N; j++) {
                world[i * N + j] += DT * growth_lenia(tmp[i * N + j]);
                world[i * N + j] = fmin(1, fmax(0, world[i * N + j]));  // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * N + j] = world[i * N + j] * 255;
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
    double stop = omp_get_wtime();
    printf("Time(full): %f s\n", stop - start);
    free(w);
    free(tmp);
    free(world);
    return 0;
}