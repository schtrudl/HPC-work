#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

#include "gifenc.h"

#define LJ_GPU 1

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

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

// GPU cell list — mirrors the CPU linked-list CellList exactly.
// Rebuild is done on the CPU: download positions, run the same head/next/pcell
// logic, then upload to device.  Force kernels traverse head[nc]->next[j] chains.
#ifndef CELL_SKIN
    #define CELL_SKIN 0.3
#endif

int g_nx, g_ny, g_n_cells;
double g_inv_cx, g_inv_cy;
unsigned int g_n = 0;

// device cell list arrays (same roles as CellList in lj_cpu.cpp)
int* d_head;  // [n_cells] linked-list head per cell, -1 = empty
int* d_next;  // [n]       next particle in same cell, -1 = end
int* d_pcell;  // [n]       current cell index per particle
double* d_ref_x;  // [n]       particle x at last rebuild
double* d_ref_y;  // [n]       particle y at last rebuild
int* d_rebuild_flag;

// host-side mirrors used during CPU rebuild
int* h_head;
int* h_next;
int* h_pcell;
double* h_ref_x;
double* h_ref_y;
Vec2* h_pos_tmp;

constexpr double gpu_skin2 = (CELL_SKIN * 0.5) * (CELL_SKIN * 0.5);

__global__ void d_cl_needs_rebuild(const Vec2* pos, const double* ref_x, const double* ref_y, unsigned int n, double bs, int* needs_rebuild) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int my_needs_rebuild = 0;
    if (i < n) {
        double dx = pos[i].x - ref_x[i];
        double dy = pos[i].y - ref_y[i];
        dx -= bs * nearbyint(dx / bs);
        dy -= bs * nearbyint(dy / bs);
        if (dx * dx + dy * dy > gpu_skin2) {
            my_needs_rebuild = 1;
        }
    }
    block_reduce(my_needs_rebuild);
    if (threadIdx.x == 0 && my_needs_rebuild) {
        *needs_rebuild = 1;
    }
}

/// Rebuild cell list on CPU, then upload it back to device.
void cl_rebuild(unsigned int n) {
    // Download current positions from device
    checkCudaErrors(cudaMemcpy(h_pos_tmp, d_pos, n * sizeof(Vec2), cudaMemcpyDeviceToHost));

    // CPU rebuild — identical logic to cl_rebuild() in lj_cpu.cpp
    for (int c = 0; c < g_n_cells; c++)
        h_head[c] = -1;
    for (int i = (int)n - 1; i >= 0; i--) {
        int cx = (int)(h_pos_tmp[i].x * g_inv_cx);
        int cy = (int)(h_pos_tmp[i].y * g_inv_cy);
        if (cx < 0)
            cx = 0;
        else if (cx >= g_nx)
            cx = g_nx - 1;
        if (cy < 0)
            cy = 0;
        else if (cy >= g_ny)
            cy = g_ny - 1;
        int c = cy * g_nx + cx;
        h_pcell[i] = c;
        h_next[i] = h_head[c];
        h_head[c] = i;
        h_ref_x[i] = h_pos_tmp[i].x;
        h_ref_y[i] = h_pos_tmp[i].y;
    }

    // Upload new cell list to device
    checkCudaErrors(cudaMemcpyAsync(d_head, h_head, g_n_cells * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_next, h_next, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_pcell, h_pcell, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_ref_x, h_ref_x, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_ref_y, h_ref_y, n * sizeof(double), cudaMemcpyHostToDevice));
}

/// the check is done on the GPU
/// because it can be done in parallel and avoids copy
bool cl_needs_rebuild(unsigned int n, double box_size) {
    checkCudaErrors(cudaMemset(d_rebuild_flag, 0, sizeof(int)));
    dim3 blk(BLOCK_SIZE);
    dim3 grd((n + blk.x - 1) / blk.x);
    d_cl_needs_rebuild<<<grd, blk>>>(d_pos, d_ref_x, d_ref_y, n, box_size, d_rebuild_flag);
    checkCudaErrors(cudaGetLastError());
    int flag = 0;
    checkCudaErrors(cudaMemcpy(&flag, d_rebuild_flag, sizeof(int), cudaMemcpyDeviceToHost));
    return flag != 0;
}

void init_cuda(Particle* particles, unsigned int n, double box_size) {
    checkCudaErrors(cudaMalloc((void**)&d_pos, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_vel, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec2)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));

    // Set up cell list params (same formula as lj_cpu.cpp)
    double cell_size = R_CUT + CELL_SKIN;
    g_nx = (int)(box_size / cell_size);
    g_ny = (int)(box_size / cell_size);
    if (g_nx < 3) g_nx = 3;
    if (g_ny < 3) g_ny = 3;
    g_n_cells = g_nx * g_ny;
    g_inv_cx = (double)g_nx / box_size;
    g_inv_cy = (double)g_ny / box_size;

    checkCudaErrors(cudaMalloc((void**)&d_head, g_n_cells * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_next, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_pcell, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_ref_x, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_ref_y, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_rebuild_flag, sizeof(int)));

    h_head = (int*)malloc(g_n_cells * sizeof(int));
    h_next = (int*)malloc(n * sizeof(int));
    h_pcell = (int*)malloc(n * sizeof(int));
    h_ref_x = (double*)malloc(n * sizeof(double));
    h_ref_y = (double*)malloc(n * sizeof(double));
    h_pos_tmp = (Vec2*)malloc(n * sizeof(Vec2));

    g_n = n;
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

/// Cell-list based GPU force kernel with PE.
/// One thread per particle i; traverses head[nc]->next[j] chain per neighbour cell.
__global__ void d_compute_forces(Vec2* pos, Vec2* force, unsigned int n, double box_size, double* result, const int* head, const int* next, const int* pcell, int nx, int ny) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec2 pi = pos[i];
    double fix = 0.0, fiy = 0.0, pe = 0.0;

    int ci = pcell[i];
    int cix = ci % nx;
    int ciy = ci / nx;

    for (int ndy = -1; ndy <= 1; ndy++) {
        int ncy = (ciy + ndy + ny) % ny;
        for (int ndx = -1; ndx <= 1; ndx++) {
            int ncx = (cix + ndx + nx) % nx;
            int nc = ncy * nx + ncx;
            for (int j = head[nc]; j >= 0; j = next[j]) {
                if (j == (int)i) continue;
                Vec2 pj = pos[j];
                double dx = pi.x - pj.x;
                double dy = pi.y - pj.y;
                dx -= box_size * nearbyint(dx / box_size);
                dy -= box_size * nearbyint(dy / box_size);
                double r2 = dx * dx + dy * dy;
                if (r2 >= rc2 || r2 == 0.0) continue;
                double sr2 = (SIGMA * SIGMA) / r2;
                double sr6 = sr2 * sr2 * sr2;
                double sr12 = sr6 * sr6;
                double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                fix += fij * dx;
                fiy += fij * dy;
                pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
            }
        }
    }

    force[i].x = fix;
    force[i].y = fiy;
    atomicAdd(result, pe);
}

__global__ void d_compute_forces(Vec2* __restrict__ pos, Vec2* __restrict__ force, unsigned int n, double box_size, double* result) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Vec2 s_pos[BLOCK_SIZE];

    double fix = 0.0, fiy = 0.0, pe = 0.0;
    Vec2 pi = (i < n) ? pos[i] : make_double2(0.0, 0.0);

    for (unsigned int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        unsigned int j_load = tile * BLOCK_SIZE + threadIdx.x;
        s_pos[threadIdx.x] = (j_load < n) ? pos[j_load] : make_double2(0.0, 0.0);
        __syncthreads();

        if (i < n) {
            for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
                unsigned int j = tile * BLOCK_SIZE + k;
                if (j >= n || j == i) continue;
                Vec2 pj = s_pos[k];
                double dx = pi.x - pj.x;
                double dy = pi.y - pj.y;
                dx -= box_size * nearbyint(dx / box_size);
                dy -= box_size * nearbyint(dy / box_size);
                double r2 = dx * dx + dy * dy;
                if (r2 < rc2 && r2 != 0.0) {
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij * dx;
                    fiy += fij * dy;
                    pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }
        __syncthreads();
    }

    if (i < n) {
        force[i].x = fix;
        force[i].y = fiy;
    }
    pe = block_reduce(pe);
    if (threadIdx.x == 0) atomicAdd(result, pe);
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

//restrict pove compilerju da se dva kazalca v pomnilniku ne prekrivata, to omogoča boljše optimizacije 
__global__ void d_compute_forces_no_pe(Vec2* __restrict__ pos, Vec2* __restrict__ force, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Vec2 s_pos[BLOCK_SIZE];

    double fix = 0.0, fiy = 0.0;
    Vec2 pi = (i < n) ? pos[i] : make_double2(0.0, 0.0);

    for (unsigned int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++){
        unsigned int j_load = tile * BLOCK_SIZE + threadIdx.x;
        s_pos[threadIdx.x] = (j_load < n) ? pos[j_load] : make_double2(0.0,0.0);
        __syncthreads();

        if (i < n){
            for (unsigned int k = 0; k < BLOCK_SIZE; k++){
                unsigned int j = tile * BLOCK_SIZE + k;
                if (j >= n || j == i) continue;
                Vec2 pj = s_pos[k];
                double dx = pi.x - pj.x;
                double dy = pi.y - pj.y;
                dx -= box_size * nearbyint(dx / box_size);
                dy -= box_size * nearbyint(dy / box_size);
                double r2 = dx * dx + dy * dy;
                if (r2 < rc2 && r2 != 0.0){
                    double sr2 = (SIGMA * SIGMA) /r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij * dx;
                    fiy += fij * dy;
                }
            }
        }
        __syncthreads();
    }

    if (i < n){
        force[i].x = fix;
        force[i].y = fiy;
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

    // each thread for one particle
    dim3 block_size_n(BLOCK_SIZE);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);

    // 2D grid for n**2
    //dim3 block_size_2d(BLOCK_SIZE, 1);
    //dim3 grid_size_2d((n - 1) / block_size_2d.x + 1, (n - 1) / block_size_2d.y + 1);

    checkCudaErrors(cudaMemcpyAsync(d_pos, particles[0].position, n * sizeof(Vec2), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_vel, particles[0].velocity, n * sizeof(Vec2), cudaMemcpyHostToDevice));

    cl_rebuild(n);

    // forces are zeroed on initialize_particles
    // result is zeroed from init
    d_compute_forces<<<grid_size_n, block_size_n>>>(d_pos, d_force, n, box_size, d_result);
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
        if (cl_needs_rebuild(n, box_size)) {
            cl_rebuild(n);
        }
        checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
        d_compute_forces_no_pe<<<grid_size_n, block_size_n>>>(d_pos, d_force, n, box_size);
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
        if (cl_needs_rebuild(n, box_size)) {
            cl_rebuild(n);
        }
        checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
        checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec2)));
        d_compute_forces<<<grid_size_n, block_size_n>>>(d_pos, d_force, n, box_size, d_result);
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
