#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "orbium.h"
#include "gifenc.h"

// Include CUDA headers
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "const_kernel.h"

// Uncomment to generate gif animation
// #define GENERATE_GIF

#ifndef SIZE
    #define SIZE 64
#endif
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 32
#endif
#ifndef BLOCK_SIZE_X
    #define BLOCK_SIZE_X BLOCK_SIZE
#endif
#ifndef BLOCK_SIZE_Y
    #define BLOCK_SIZE_Y BLOCK_SIZE
#endif

#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define HALO (KERNEL_SIZE / 2)  // 13
#define NUM_ORBIUMS 2

// For prettier indexing syntax
#define w(r, c) (w[(r) * KERNEL_SIZE + (c)])
#define input(r, c) (world[((r) % SIZE) * SIZE + ((c) % SIZE)])
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
__device__ __forceinline__ f32 gauss(const f32 x, const f32 mu, const f32 sigma) {
    f32 z = (x - mu) / sigma;
    return expf(-0.5f * z * z);
}

// Function for growth criteria
__device__ __forceinline__ f32 growth_lenia(const f32 u) {
    f32 mu = 0.15f;
    f32 sigma = 0.015f;
    return -1.0f + 2.0f * gauss(u, mu, sigma);  // Baseline -1, peak +1
}

// Polynomial approximation of growth_lenia (max error: 1.26e-05)
__device__ __forceinline__ f32 growth_lenia_poly(const f32 u) {
    const f32 mu = 0.15f, sigma = 0.015f;
    if (u < 0.09f || u > 0.21f) return -1.0f;
    const f32 t = (u - mu) / sigma, t2 = t * t;
    return ((((((((((1.5164566482e-11f) * t2 + (-1.5024997416e-09f)) * t2 + (6.7570921873e-08f)) * t2 + (-1.8452074685e-06f)) * t2 + (3.4634562942e-05f)) * t2 + (-4.7938982646e-04f)) * t2 + (5.0822995568e-03f)) * t2 + (-4.1438623715e-02f)) * t2 + (2.4978575127e-01f)) * t2 + (-9.9992089585e-01f)) * t2 + (9.9999515209e-01f);
}

__global__ void whole_step(f32* world, f32* new_world, int tile_w, int tile_h) {
    extern __shared__ f32 world_tile[];  // tile_h * tile_w

    const int block_y = blockIdx.y * blockDim.y;
    const int block_x = blockIdx.x * blockDim.x;

    // Load entire tile
    for (int y = threadIdx.y; y < tile_h; y += blockDim.y) {
        for (int x = threadIdx.x; x < tile_w; x += blockDim.x) {
            int gy = (block_y - HALO + y + SIZE) % SIZE;
            int gx = (block_x - HALO + x + SIZE) % SIZE;
            world_tile[y * tile_w + x] = world[gy * SIZE + gx];
        }
    }
    __syncthreads();

    int x = block_x + threadIdx.x;
    int y = block_y + threadIdx.y;
    if (x >= SIZE || y >= SIZE) return;

    f32 sum = 0;
    for (int k = 0; k < NUM_KERNEL_ENTRIES; k++) {
        int ty = HALO + threadIdx.y + d_sparse_k[k].dy;
        int tx = HALO + threadIdx.x + d_sparse_k[k].dx;
        sum += d_sparse_k[k].weight * world_tile[ty * tile_w + tx];
    }
    int idx = y * SIZE + x;
    new_world[idx] = __saturatef(world[idx] + DT * growth_lenia(sum));
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

    float time;

    // Allocate memory
    f32* world = (f32*)calloc(SIZE * SIZE, sizeof(f32));
    f32* tmp = (f32*)calloc(SIZE * SIZE, sizeof(f32));

    // Place orbiums
    for (unsigned int o = 0; o < NUM_ORBIUMS; o++) {
        world = place_orbium(world, SIZE, SIZE, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    cudaEvent_t start_compute, stop_compute;
    checkCudaErrors(cudaEventCreate(&start_compute));
    checkCudaErrors(cudaEventCreate(&stop_compute));
    init_kernel_const();

    f32 *d_buffer, *d_buffer2;
    checkCudaErrors(cudaMalloc((void**)&d_buffer, (SIZE) * (SIZE) * sizeof(f32)));
    checkCudaErrors(cudaMalloc((void**)&d_buffer2, (SIZE) * (SIZE) * sizeof(f32)));

    dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const int tile_w = blockSize.x + KERNEL_SIZE;
    const int tile_h = blockSize.y + KERNEL_SIZE;
    const size_t tile_mem_size = tile_w * tile_h * sizeof(f32);
    // (next) multiple of block size to cover the whole grid
    /*
>>> BLOCK_SIZE = 16
>>> N = 63
>>> (N-1)//BLOCK_SIZE+1
4
>>> (N+BLOCK_SIZE-1)//BLOCK_SIZE
4
>>> N = 64
>>> (N+BLOCK_SIZE-1)//BLOCK_SIZE
4
>>> (N-1)//BLOCK_SIZE+1
4
>>> N = 65
>>> (N+BLOCK_SIZE-1)//BLOCK_SIZE
5
>>> (N-1)//BLOCK_SIZE+1
5
    */
    dim3 gridSize((SIZE - 1) / blockSize.x + 1, (SIZE - 1) / blockSize.y + 1);

    checkCudaErrors(cudaEventRecord(start));
    checkCudaErrors(cudaMemcpyAsync(d_buffer, world, (SIZE) * (SIZE) * sizeof(f32), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start_compute));

    // Lenia Simulation
    for (unsigned int step = 0; step < NUM_STEPS; step++) {
        // Convolution + Evolution fused
        whole_step<<<gridSize, blockSize, tile_mem_size>>>(d_buffer, d_buffer2, tile_w, tile_h);
        checkCudaErrors(cudaGetLastError());
        // Swap buffers
        f32* temp = d_buffer;
        d_buffer = d_buffer2;
        d_buffer2 = temp;

#ifdef GENERATE_GIF
        checkCudaErrors(cudaMemcpy(world, d_buffer, (SIZE) * (SIZE) * sizeof(f32), cudaMemcpyDeviceToHost));
        for (usize i = 0; i < SIZE; i++) {
            for (usize j = 0; j < SIZE; j++) {
                gif->frame[i * SIZE + j] = world[i * SIZE + j] * 255;
            }
        }
        ge_add_frame(gif, 5);
#endif
    }

    checkCudaErrors(cudaEventRecord(stop_compute));
#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
    checkCudaErrors(cudaMemcpy(world, d_buffer, (SIZE) * (SIZE) * sizeof(f32), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time, start_compute, stop_compute));
    printf("Time(compute): %f s\n", time / 1000.0);
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Time(full): %f s\n", time / 1000.0);
    cudaFree(d_buffer);
    cudaFree(d_buffer2);
    free(tmp);
    free(world);
    return 0;
}