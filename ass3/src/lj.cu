#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

#include "gifenc.h"
#ifndef LJ_GPU
    #define LJ_GPU 1
#endif
#if LJ_GPU
    // Include CUDA headers
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include "helper_cuda.h"
#endif
#include "vec2-lennard-jones.h"

#define usize size_t
#define WARP_SIZE 32
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 256
#endif

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
#if LJ_GPU
    unsigned int n_ceil_2d = (n + 31) / 32;
    //checkCudaErrors(cudaMalloc((void**)&d_particles, n * sizeof(Particle)));
    checkCudaErrors(cudaMalloc((void**)&d_pos, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_vel, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
#endif
}

// TODO(perf): I think this method is not measured, so we need to do as much work here as possible (all?)
int initialize_particles(Particle* particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    srand(seed);
    Vec2* pos = (Vec2*)calloc(n, sizeof(Vec2));
    Vec2* vel = (Vec2*)calloc(n, sizeof(Vec2));
    Vec2* force = (Vec2*)calloc(n, sizeof(Vec2));
    particles[0].position = pos;
    particles[0].velocity = vel;
    particles[0].force = force;
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
constexpr double sigma_over_rcut = SIGMA / R_CUT;
constexpr double sigma_over_rcut_2 = sigma_over_rcut * sigma_over_rcut;
constexpr double sigma_over_rcut_6 = sigma_over_rcut_2 * sigma_over_rcut_2 * sigma_over_rcut_2;
constexpr double sigma_over_rcut_12 = sigma_over_rcut_6 * sigma_over_rcut_6;
constexpr double v_shift = 4.0 * EPSILON * (sigma_over_rcut_12 - sigma_over_rcut_6);

constexpr double rc2 = R_CUT * R_CUT;

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
            double r2 = dx * dx + dy * dy;
            if (r2 >= rc2 || r2 == 0.0) {
                continue;
            }
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            double fx = fij * dx;
            double fy = fij * dy;

            fi->x += fx;
            fi->y += fy;

            double vij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
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
        double r2 = dx * dx + dy * dy;
        if (r2 < rc2 && r2 != 0.0) {
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            fx = fij * dx;
            fy = fij * dy;

            double vij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
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
            double r2 = dx * dx + dy * dy;
            if (r2 >= rc2 || r2 == 0.0) {
                continue;
            }
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            double fx = fij * dx;
            double fy = fij * dy;

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
        double r2 = dx * dx + dy * dy;
        if (r2 < rc2 && r2 != 0.0) {
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            fx = fij * dx;
            fy = fij * dy;
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
    SimulationResult out;
#if LJ_GPU

    // each thread for one particle
    dim3 block_size_n(BLOCK_SIZE);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);

    // 2D grid for n**2
    dim3 block_size_2d(BLOCK_SIZE, 1);
    dim3 grid_size_2d((n - 1) / block_size_2d.x + 1, (n - 1) / block_size_2d.y + 1);

    checkCudaErrors(cudaMemcpyAsync(d_pos, particles[0].position, n * sizeof(Vec2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_vel, particles[0].velocity, n * sizeof(Vec2), cudaMemcpyHostToDevice));
    // forces are zeroed on initialize_particles
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
#else
    Vec2* pos = particles[0].position;
    Vec2* vel = particles[0].velocity;
    Vec2* force = particles[0].force;
    out.start_potential = compute_forces(pos, force, n, box_size);
    out.start_kinetic = compute_ke(vel, n);
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
        first_update(pos, vel, force, n, box_size);
        compute_forces_no_pe(pos, force, n, box_size);
        second_update(vel, force, n);
    #if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
    #endif
    }

    for (unsigned int step = steps_without_log; step < nsteps; step++) {
        first_update(pos, vel, force, n, box_size);
        out.final_potential = compute_forces(pos, force, n, box_size);
        second_update(vel, force, n);

        out.final_kinetic = compute_ke(vel, n);
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
#endif

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
