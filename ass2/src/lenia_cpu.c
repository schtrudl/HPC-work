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

#ifndef N
    #define N 256
#endif

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// For prettier indexing syntax
#define w(r, c) (w[(r) * KERNEL_SIZE + (c)])
#define input(r, c) (world[((r) % N) * N + ((c) % N)])
#define f64 double
#define usize size_t
#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

struct orbium_coo {
    int row;
    int col;
    int angle;
};

// Function to calculate Gaussian
inline f64 gauss(const f64 x, const f64 mu, const f64 sigma) {
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
f64 growth_lenia(const f64 u) {
    f64 mu = 0.15;
    f64 sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);  // Baseline -1, peak +1
}

// Function to generate convolution kernel
f64* generate_kernel(f64* const K, const usize size) {
    // Construct ring convolution filter
    const f64 mu = 0.5;
    const f64 sigma = 0.15;
    const int r = size / 2;
    f64 sum = 0;
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                f64 distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1) {
                    K[(y + r) * size + x + r] = 0;  // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (usize y = 0; y < size; y++) {
            for (usize x = 0; x < size; x++) {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
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
    f64* w = (f64*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(f64));
    f64* world = (f64*)calloc(N * N, sizeof(f64));
    f64* tmp = (f64*)calloc(N * N, sizeof(f64));

    // Generate convolution kernel
    w = generate_kernel(w, KERNEL_SIZE);

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        world = place_orbium(world, N, N, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    f64 start = omp_get_wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        // Convolution
        // Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
        OMP(parallel for)
        for (usize i = 0; i < N; i++) {
            for (usize j = 0; j < N; j++) {
                f64 sum = 0;
                for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        sum += w(ki, kj) * input((i - KERNEL_SIZE / 2 + N + kri), (j - KERNEL_SIZE / 2 + N + kcj));
                    }
                }
                tmp[i * N + j] = sum;
            }
        }

        // Evolution
        OMP(parallel for)
        for (unsigned int i = 0; i < N; i++) {
            for (unsigned int j = 0; j < N; j++) {
                double t = world[i * N + j];
                t += DT * growth_lenia(tmp[i * N + j]);
                world[i * N + j] = fmin(1, fmax(0, t));  // Clip between 0 and 1
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
    f64 stop = omp_get_wtime();
    printf("Time(full): %f s\n", stop - start);
    free(w);
    free(tmp);
    free(world);
    return 0;
}