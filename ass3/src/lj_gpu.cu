#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#include "gifenc.h"
#include "vec2-lennard-jones.h"

#define usize size_t
#define WARP_SIZE 32

// plotting functions
#if GENERATE_GIF
uint8_t palette[] = {0, 0, 0, 255, 255, 0};

void set_pixel(uint8_t* img, int w, int h, int x, int y, uint8_t index) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return;
    }
    size_t idx = (size_t)y * (size_t)w + (size_t)x;
    img[idx] = index;
}

void render_frame_gif(ge_GIF* gif, const Particle* particles, unsigned int n, double box_size) {
    memset(gif->frame, 0, FRAME_WIDTH * FRAME_HEIGHT);

    for (unsigned int i = 0; i < n; ++i) {
        int px = (int)(particles[0].position[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[0].position[i].y / box_size * (double)(FRAME_HEIGHT - 1));
        py = (FRAME_HEIGHT - 1) - py;

        for (int dy = -FRAME_PARTICLE_RADIUS; dy <= FRAME_PARTICLE_RADIUS; ++dy) {
            for (int dx = -FRAME_PARTICLE_RADIUS; dx <= FRAME_PARTICLE_RADIUS; ++dx) {
                if (dx * dx + dy * dy <= FRAME_PARTICLE_RADIUS * FRAME_PARTICLE_RADIUS) {
                    set_pixel(gif->frame, FRAME_WIDTH, FRAME_HEIGHT, px + dx, py + dy, 1);
                }
            }
        }
    }
}
#endif
double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double inline compute_ke(const Vec2* velocity, unsigned int n) {
    double ke = 0.0;
    OMP(parallel for reduction(+:ke))
    for (unsigned int i = 0; i < n; ++i) {
        const Vec2 v = velocity[i];
        ke += 0.5 * (v.x * v.x + v.y * v.y);
    }
    return ke;
}

// special function that performs warp-level reduction to sum
// using shuffle instructions, which is more efficient than shared memory reduction
__inline__ __device__ double warp_reduce(double val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
__device__ __inline__ double block_reduce(double val) {
    // max 1024 threads per block / 32 threads per warp = 32 warps
    static __shared__ double sdata[WARP_SIZE];  // shared memory for warp-level reductions
    int lane = threadIdx.x % WARP_SIZE;  // thread id in warp
    int warp = threadIdx.x / WARP_SIZE;  // warp id in block

    val = warp_reduce(val);  // each warp reduces to a single value

    if (lane == 0) {
        sdata[warp] = val;
    }
    __syncthreads();  // await all warps

    // reduce the values from each warp using the first warp
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? sdata[lane] : 0.0;
    if (warp == 0) {
        val = warp_reduce(val);  // final reduction within the first warp
    }
    return val;  // only thread 0 will have the final result
}

__global__ void d_compute_ke(const Vec2* velocity, unsigned int n, double* result) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double ke = 0.0;
    if (i < n) {
        const Vec2 v = velocity[i];
        ke = 0.5 * (v.x * v.x + v.y * v.y);
    }

    ke = block_reduce(ke);
    if (threadIdx.x == 0) atomicAdd(result, ke);
}

Vec2* d_pos;
Vec2* d_vel;
Vec2* d_force;
double* d_result;

void init_cuda(Particle* particles, unsigned int n, double box_size) {
    unsigned int n_ceil_2d = (n + 31) / 32;
    //checkCudaErrors(cudaMalloc((void**)&d_particles, n * sizeof(Particle)));
    checkCudaErrors(cudaMalloc((void**)&d_pos, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_vel, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
}

// TODO(perf): I think this method is not measured, so we need to do as much work here as possible (all?)
int initialize_particles(Particle* particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    srand(seed);
    Vec2* pos = (Vec2*)calloc(n, sizeof(Vec2));
    Vec2* vel = (Vec2*)calloc(n, sizeof(Vec2));
    Vec2* force = (Vec2*)calloc(n, sizeof(Vec2));
    unsigned int n_side = (unsigned int)ceil(sqrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
    // this loop should not be parallelized to ensure reproducibility of the initial conditions (due to rand and mean)
    for (unsigned int k = 0; k < n; k++) {
        double x0 = offset + (0.5 + (double)(k % n_side)) * delta;
        double y0 = offset + (0.5 + (double)(k / n_side)) * delta;

        pos[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        pos[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        vel[k].x = 2.0 * random_double() - 1.0;
        vel[k].y = 2.0 * random_double() - 1.0;

        mean_vx += vel[k].x;
        mean_vy += vel[k].y;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        vel[k].x -= mean_vx;
        vel[k].y -= mean_vy;
        ke += 0.5 * (vel[k].x * vel[k].x + vel[k].y * vel[k].y);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        init_cuda(particles, n, box_size);
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        vel[k].x *= scale;
        vel[k].y *= scale;
    }

    particles[0].position = pos;
    particles[0].velocity = vel;
    particles[0].force = force;

    init_cuda(particles, n, box_size);
    return 1;
}

void inline first_update(Vec2* pos, Vec2* vel, Vec2* force, unsigned int n, double box_size) {
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        Vec2* p_pos = &pos[i];
        Vec2* p_vel = &vel[i];
        const Vec2 p_force = force[i];
        p_vel->x += 0.5 * DT * p_force.x;
        p_vel->y += 0.5 * DT * p_force.y;

        // fused wrap_positions
        // apply periodic boundary conditions to ensure particles stay within the simulation box
        double wx = fmod(p_pos->x + DT * p_vel->x, box_size);
        double wy = fmod(p_pos->y + DT * p_vel->y, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p_pos->x = wx;
        p_pos->y = wy;
    }
}

__global__ void d_first_update(Vec2* pos, Vec2* vel, Vec2* force, unsigned int n, double box_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec2* p_pos = &pos[i];
    Vec2* p_vel = &vel[i];
    const Vec2 p_force = force[i];
    p_vel->x += 0.5 * DT * p_force.x;
    p_vel->y += 0.5 * DT * p_force.y;

    // fused wrap_positions
    // apply periodic boundary conditions to ensure particles stay within the simulation box
    double wx = fmod(p_pos->x + DT * p_vel->x, box_size);
    double wy = fmod(p_pos->y + DT * p_vel->y, box_size);

    if (wx < 0.0) {
        wx += box_size;
    }
    if (wy < 0.0) {
        wy += box_size;
    }

    p_pos->x = wx;
    p_pos->y = wy;
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
const double v_shift = 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));

__device__ __forceinline__ double compute_v_shift(void) {
    return 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));
}

double inline compute_forces(Vec2* pos, Vec2* force, unsigned int n, double box_size) {
    double pe = 0.0;
    OMP(parallel for reduction(+:pe))
    for (unsigned int i = 0; i < n; ++i) {
        const Vec2 pi = pos[i];
        Vec2* fi = &force[i];
        fi->x = 0.0;
        fi->y = 0.0;
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const Vec2 pj = pos[j];

            // compute distance between particles with periodic boundary conditions
            double dx = pi.x - pj.x;
            double dy = pi.y - pj.y;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;

            fi->x += fx;
            fi->y += fy;

            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    return pe;
}

__global__ void d_compute_forces(Vec2* pos, Vec2* force, unsigned int n, double box_size, double* result) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    double pe = 0.0, fx = 0.0, fy = 0.0;
    if (i < n && j < n && i != j) {
        Vec2 pi = pos[i];
        Vec2 pj = pos[j];

        // compute distance between particles with periodic boundary conditions
        double dx = pi.x - pj.x;
        double dy = pi.y - pj.y;

        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);

        // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
        double r = sqrt(dx * dx + dy * dy);
        if (r < R_CUT && r != 0.0) {
            double sr = SIGMA / r;
            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            fx = fij * dx / r;
            fy = fij * dy / r;

            double d_v_shift = compute_v_shift();
            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - d_v_shift;
            pe = 0.5 * vij;  // (j,i) thread contributes the other 0.5
        }
    }

    fx = block_reduce(fx);
    fy = block_reduce(fy);
    pe = block_reduce(pe);
    if (threadIdx.x == 0) {
        atomicAdd(&force[i].x, fx);
        atomicAdd(&force[i].y, fy);
        atomicAdd(result, pe);
    }
}

void inline compute_forces_no_pe(Vec2* pos, Vec2* force, unsigned int n, double box_size) {
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        const Vec2 pi = pos[i];
        Vec2* fi = &force[i];
        fi->x = 0.0;
        fi->y = 0.0;
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            const Vec2 pj = pos[j];

            // compute distance between particles with periodic boundary conditions
            double dx = pi.x - pj.x;
            double dy = pi.y - pj.y;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;

            fi->x += fx;
            fi->y += fy;
        }
    }
}

__global__ void d_compute_forces_no_pe(Vec2* pos, Vec2* force, unsigned int n, double box_size) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    double fx = 0.0, fy = 0.0;
    if (i < n && j < n && i != j) {
        const Vec2 pi = pos[i];
        const Vec2 pj = pos[j];

        // compute distance between particles with periodic boundary conditions
        double dx = pi.x - pj.x;
        double dy = pi.y - pj.y;

        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);

        // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
        double r = sqrt(dx * dx + dy * dy);
        if (r < R_CUT && r != 0.0) {
            double sr = SIGMA / r;
            double fij = 24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            fx = fij * dx / r;
            fy = fij * dy / r;
        }
    }

    fx = block_reduce(fx);
    fy = block_reduce(fy);
    if (threadIdx.x == 0) {
        atomicAdd(&force[i].x, fx);
        atomicAdd(&force[i].y, fy);
    }
}

void inline second_update(Vec2* vel, Vec2* force, unsigned int n) {
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        Vec2* v = &vel[i];
        const Vec2 f = force[i];
        v->x += 0.5 * DT * f.x;
        v->y += 0.5 * DT * f.y;
    }
}

__global__ void d_second_update(Vec2* vel, Vec2* force, unsigned int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vec2* v = &vel[i];
    const Vec2 f = force[i];
    v->x += 0.5 * DT * f.x;
    v->y += 0.5 * DT * f.y;
}

SimulationResult run_simulation(Particle* particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    // each thread for one particle
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);

    // 2D grid: each thread handles one (i,j) pair without any symmetry optimization
    dim3 block_size_2d(256, 1);
    dim3 grid_size_2d((n - 1) / block_size_2d.x + 1, (n - 1) / block_size_2d.y + 1);

    // TODO(perf): if we measure this we can do upload and measure KE
    //checkCudaErrors(cudaMemcpyAsync(d_particles, particles, n * sizeof(Particle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pos, particles[0].position, n * sizeof(Vec2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_vel, particles[0].velocity, n * sizeof(Vec2), cudaMemcpyHostToDevice));
    // forces are zeroed on initialize_particles
    SimulationResult out;
    //out.start_potential = compute_forces(particles, n, box_size);
    //out.start_kinetic = compute_ke(particles, n);
    // result is zeroed from init
    d_compute_forces<<<grid_size_2d, block_size_2d>>>(d_pos, d_force, n, box_size, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&out.start_potential, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    d_compute_ke<<<grid_size_n, block_size_n>>>(d_vel, n, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&out.start_kinetic, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    out.start_total = out.start_kinetic + out.start_potential;

#if GENERATE_GIF
    ge_GIF* gif = NULL;

    gif = ge_new_gif(GIF_FILE, (uint16_t)FRAME_WIDTH, (uint16_t)FRAME_HEIGHT, palette, 8, -1, 0);
    if (!gif) {
        fprintf(stderr, "Warning: failed to create GIF output %s\n", GIF_FILE);
    } else {
        render_frame_gif(gif, particles, n, box_size);
        ge_add_frame(gif, FRAME_DELAY);
    }
#endif
    unsigned int steps_without_log = log_steps ? 0 : nsteps - 1;
    for (unsigned int step = 0; step < steps_without_log; step++) {
        // leapfrog
        d_first_update<<<grid_size_n, block_size_n>>>(d_pos, d_vel, d_force, n, box_size);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
        d_compute_forces_no_pe<<<grid_size_2d, block_size_2d>>>(d_pos, d_force, n, box_size);
        checkCudaErrors(cudaGetLastError());
        d_second_update<<<grid_size_n, block_size_n>>>(d_vel, d_force, n);
        checkCudaErrors(cudaGetLastError());
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            checkCudaErrors(cudaMemcpy(particles[0].position, d_pos, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }
    for (unsigned int step = steps_without_log; step < nsteps; step++) {
        d_first_update<<<grid_size_n, block_size_n>>>(d_pos, d_vel, d_force, n, box_size);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
        checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
        d_compute_forces<<<grid_size_2d, block_size_2d>>>(d_pos, d_force, n, box_size, d_result);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(&out.final_potential, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        d_second_update<<<grid_size_n, block_size_n>>>(d_vel, d_force, n);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
        d_compute_ke<<<grid_size_n, block_size_n>>>(d_vel, n, d_result);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaMemcpy(&out.final_kinetic, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n", step, out.final_kinetic, out.final_potential, out.final_total);
        }

#if GENERATE_GIF

        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            checkCudaErrors(cudaMemcpy(particles[0].position, d_pos, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }
    checkCudaErrors(cudaMemcpy(particles[0].position, d_pos, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles[0].velocity, d_vel, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles[0].force, d_force, n * sizeof(Vec2), cudaMemcpyDeviceToHost));
    /*
    for (unsigned int step = steps_without_log; step < nsteps; step++) {
        first_update(particles, n, box_size);
        out.final_potential = compute_forces(particles, n, box_size);
        second_update(particles, n);

        out.final_kinetic = compute_ke(particles, n);
        out.final_total = out.final_kinetic + out.final_potential;
        if (log_steps) {
            printf("step=%6u  KE=%12.6f  PE=%12.6f  E=%12.6f\n", step, out.final_kinetic, out.final_potential, out.final_total);
        }

#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }
*/

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    out.n = n;
    out.particles = particles;
    return out;
}

#include "main.h"

/*
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
    */
/*
int main() {

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
        */