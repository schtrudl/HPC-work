#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "orbium.h"
#include "gifenc.h"
#include "growth_lut.h"

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>

// Uncomment to generate gif animation
// #define GENERATE_GIF

#ifndef SIZE
    #define SIZE 256
#endif

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// For prettier indexing syntax
#define w(r, c) (w[(r) * KERNEL_SIZE + (c)])
#define input(r, c) (world[((r) % SIZE) * SIZE + ((c) % SIZE)])
#define f64 double
#define f32 float
#define usize size_t
#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

struct orbium_coo {
    int row;
    int col;
    int angle;
};

// Function to calculate Gaussian
inline f32 gauss(const f32 x, const f32 mu, const f32 sigma) {
    f32 z = (x - mu) / sigma;
    return expf(-0.5f * z * z);
}

// Function for growth criteria
inline f32 growth_lenia(const f32 u) {
    // floats???
    if (u == 0.0f) return -1.0f;  // Fast path for empty cells
    const f32 mu = 0.15f;
    const f32 sigma = 0.015f;
    return -1.0f + 2.0f * gauss(u, mu, sigma);  // Baseline -1, peak +1
}

// Polynomial approximation of growth_lenia (max error: 1.26e-05)
inline f32 growth_lenia_poly(const f32 u) {
    const f32 mu = 0.15f, sigma = 0.015f;
    if (u < 0.09f || u > 0.21f) return -1.0f;
    const f32 t = (u - mu) / sigma, t2 = t * t;
    return ((((((((((1.5164566482e-11f) * t2 + (-1.5024997416e-09f)) * t2 + (6.7570921873e-08f)) * t2 + (-1.8452074685e-06f)) * t2 + (3.4634562942e-05f)) * t2 + (-4.7938982646e-04f)) * t2 + (5.0822995568e-03f)) * t2 + (-4.1438623715e-02f)) * t2 + (2.4978575127e-01f)) * t2 + (-9.9992089585e-01f)) * t2 + (9.9999515209e-01f);
}

// Function to generate convolution kernel
f32* generate_kernel(f32* const K, const usize size) {
    // Construct ring convolution filter
    const f32 mu = 0.5f;
    const f32 sigma = 0.15f;
    const int r = size / 2;
    f32 sum = 0.0f;
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                f32 distance = sqrtf((1.0f + x) * (1.0f + x) + (1.0f + y) * (1.0f + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1.0f) {
                    K[(y + r) * size + x + r] = 0.0f;  // Cut at d=1
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
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, SIZE / 3, 0}, {SIZE / 3, 0, 180}};

int main() {
#ifdef GENERATE_GIF
    ge_GIF* gif = ge_new_gif(
        "lenia.gif", /* file name */
        SIZE,
        SIZE, /* canvas size */
        (uint8_t*)inferno_pallete, /*pallete*/
        8, /* palette depth == log2(# of colors) */
        -1, /* no transparency */
        0 /* infinite loop */
    );
#endif

    // Allocate memory
    f32* w = (f32*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(f32));
    f32* world = (f32*)calloc(SIZE * SIZE, sizeof(f32));
    f32* tmp = (f32*)calloc(SIZE * SIZE, sizeof(f32));

    // Generate convolution kernel
    w = generate_kernel(w, KERNEL_SIZE);

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        world = place_orbium(world, SIZE, SIZE, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    f64 start = omp_get_wtime();

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        // Convolution
        // Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
        OMP(parallel for)
        for (usize i = 0; i < SIZE; i++) {
            for (usize j = 0; j < SIZE; j++) {
                f32 sum = 0;
                for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        sum += w(ki, kj) * input((i - KERNEL_SIZE / 2 + SIZE + kri), (j - KERNEL_SIZE / 2 + SIZE + kcj));
                    }
                }
                tmp[i * SIZE + j] = sum;
            }
        }

        // Evolution
        OMP(parallel for)
        for (unsigned int i = 0; i < SIZE; i++) {
            for (unsigned int j = 0; j < SIZE; j++) {
                f32 t = world[i * SIZE + j];
                t += DT * growth_lenia_poly(tmp[i * SIZE + j]);
                world[i * SIZE + j] = fminf(1, fmaxf(0, t));  // Clip between 0 and 1
#ifdef GENERATE_GIF
                gif->frame[i * SIZE + j] = world[i * SIZE + j] * 255;
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