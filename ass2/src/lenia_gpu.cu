#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "orbium.h"
#include "gifenc.h"

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>

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
    return expf(-0.5f * powf((x - mu) / sigma, 2));
}

// Function for growth criteria
f32 growth_lenia(const f32 u) {
    f32 mu = 0.15f;
    f32 sigma = 0.015f;
    return -1.0f + 2.0f * gauss(u, mu, sigma);  // Baseline -1, peak +1
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

__global__ void conv_step(f32* world, f32* w, f32* tmp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int idx = j * N + i;

    f32 sum = 0;
    for (int ki = KERNEL_SIZE - 1, kri = 0; ki >= 0; ki--, kri++) {
        for (int kj = KERNEL_SIZE - 1, kcj = 0; kj >= 0; kj--, kcj++) {
            sum += w(ki, kj) * input((i - KERNEL_SIZE / 2 + N + kri), (j - KERNEL_SIZE / 2 + N + kcj));
        }
    }
    tmp[idx] = sum;
}

__global__ void next_step(f32* world, f32* tmp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int idx = j * N + i;

    f32 t = world[i * N + j];
    t += DT * growth_lenia(tmp[i * N + j]);
    world[i * N + j] = fmin(1, fmax(0, t));
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
    f32* w = (f32*)calloc(KERNEL_SIZE * KERNEL_SIZE, sizeof(f32));
    f32* world = (f32*)calloc(N * N, sizeof(f32));
    f32* tmp = (f32*)calloc(N * N, sizeof(f32));

    // Generate convolution kernel
    w = generate_kernel(w, KERNEL_SIZE);

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        world = place_orbium(world, N, N, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    f32 d_w[KERNEL_SIZE * KERNEL_SIZE];
    checkCudaErrors(cudaMemcpy(d_w, w, KERNEL_SIZE * KERNEL_SIZE * sizeof(f32), cudaMemcpyHostToDevice));

    f32 *d_buffer, *d_buffer2;
    checkCudaErrors(cudaMalloc((void**)&d_buffer, (N) * (N) * sizeof(f32)));
    checkCudaErrors(cudaMalloc((void**)&d_buffer2, (N) * (N) * sizeof(f32)));

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    checkCudaErrors(cudaEventRecord(start));

    checkCudaErrors(cudaMemcpy(d_buffer, world, (N) * (N) * sizeof(f32), cudaMemcpyHostToDevice));

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        // Convolution
        conv_step<<<gridSize, blockSize>>>(d_buffer, d_w, d_buffer2);
        checkCudaErrors(cudaGetLastError());

        // Evolution
        next_step<<<gridSize, blockSize>>>(d_buffer, d_buffer2);
        checkCudaErrors(cudaGetLastError());

#ifdef GENERATE_GIF
        checkCudaErrors(cudaMemcpy(world, d_buffer, (N) * (N) * sizeof(f32), cudaMemcpyDeviceToHost));
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
checkCudaErrors(cudaMemcpy(world, d_buffer, (N) * (N) * sizeof(f32), cudaMemcpyDeviceToHost));
checkCudaErrors(cudaEventRecord(stop));
checkCudaErrors(cudaEventSynchronize(stop));
float time;
checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
printf("Time(full): %f s\n", time / 1000.0);
free(w);
free(tmp);
free(world);
return 0;
}