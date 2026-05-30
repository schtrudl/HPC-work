#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cuda.h>
#define WARP_SIZE 32

#define Vec3 double3

#define DEBUG 1
#if DEBUG
    #define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
    #define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
#else
    #define checkCudaErrors(val) (val)
    #define getLastCudaError(msg) ((void)0)
#endif

void check(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, line, (int)result, cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

inline void __getLastCudaError(const char* errorMessage, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(
            stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file,
            line,
            errorMessage,
            (int)(err),
            cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#include "lennard-jones.h"

double random_double(void) {
    return (double)rand() / (double)RAND_MAX;
}

// compute kinetic energy of the system
double compute_ke(const Particle* particles, unsigned int n) {
    double ke = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        const Particle* p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy + p->vz * p->vz);
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

__global__ void d_compute_ke(const Vec3* velocity, unsigned int n, double* result) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double ke = 0.0;
    if (i < n) {
        const Vec3 v = velocity[i];
        ke = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    ke = block_reduce(ke);
    if (threadIdx.x == 0) atomicAdd(result, ke);
}

inline double g_compute_ke(Vec3* velocity, unsigned int n, double* d_result) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    d_compute_ke<<<grid_size_n, block_size_n>>>(velocity, n, d_result);
    checkCudaErrors(cudaGetLastError());

    double ke;
    checkCudaErrors(cudaMemcpy(&ke, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return ke;
}

__global__ void d_first_update(Vec3* positions, Vec3* velocities, Vec3* forces, unsigned int n, double box_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const Vec3 force = forces[i];
    Vec3 vel = velocities[i];
    vel.x += 0.5 * DT * force.x;
    vel.y += 0.5 * DT * force.y;
    vel.z += 0.5 * DT * force.z;
    velocities[i] = vel;

    // fused wrap_positions
    Vec3 pos = positions[i];
    // apply periodic boundary conditions to ensure particles stay within the simulation box
    pos.x = fmod(pos.x + DT * vel.x, box_size);
    pos.y = fmod(pos.y + DT * vel.y, box_size);
    pos.z = fmod(pos.z + DT * vel.z, box_size);

    if (pos.x < 0.0) {
        pos.x += box_size;
    }
    if (pos.y < 0.0) {
        pos.y += box_size;
    }
    if (pos.z < 0.0) {
        pos.z += box_size;
    }

    positions[i] = pos;
}

inline void g_first_update(Vec3* pos, Vec3* vel, Vec3* force, unsigned int n, double box_size) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    d_first_update<<<grid_size_n, block_size_n>>>(pos, vel, force, n, box_size);
    checkCudaErrors(cudaGetLastError());
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
constexpr double sigma_over_rcut = SIGMA / R_CUT;
constexpr double sigma_over_rcut_2 = sigma_over_rcut * sigma_over_rcut;
constexpr double sigma_over_rcut_6 = sigma_over_rcut_2 * sigma_over_rcut_2 * sigma_over_rcut_2;
constexpr double sigma_over_rcut_12 = sigma_over_rcut_6 * sigma_over_rcut_6;
constexpr double v_shift = 4.0 * EPSILON * (sigma_over_rcut_12 - sigma_over_rcut_6);

double compute_v_shift() {
    return v_shift;
}

constexpr double rc2 = R_CUT * R_CUT;

__global__ void d_compute_forces(const Vec3* position, Vec3* force, unsigned int n, double box_size, double* result) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;

    double pe = 0.0;
    Vec3 f = {0.0, 0.0, 0.0};
    if (i < n && j < n && i != j) {
        const Vec3 pi = position[i];
        const Vec3 pj = position[j];

        // compute distance between particles with periodic boundary conditions
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;

        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        dz -= box_size * nearbyint(dz / box_size);

        // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2 && r2 != 0.0) {
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double fij = -24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            f.x += fij * dx;
            f.y += fij * dy;
            f.z += fij * dz;

            double vij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
            pe = 0.5 * vij;
        }
    }

    f.x = block_reduce(f.x);
    f.y = block_reduce(f.y);
    f.z = block_reduce(f.z);
    pe = block_reduce(pe);
    if (threadIdx.x == 0) {
        atomicAdd(&force[i].x, f.x);
        atomicAdd(&force[i].y, f.y);
        atomicAdd(&force[i].z, f.z);
        atomicAdd(result, pe);
    }
}

inline double g_compute_forces(Vec3* position, Vec3* force, unsigned int n, double box_size, double* d_result) {
    dim3 block_size_n(256, 1);
    unsigned int grid_y = ((n - 1) / block_size_n.y + 1);
    unsigned int grid_y_clamped = (grid_y > 65535U) ? 65535U : grid_y;
    unsigned int grid_z = (grid_y + grid_y_clamped - 1) / grid_y_clamped;
    dim3 grid_size_n((n - 1) / block_size_n.x + 1, grid_y_clamped, grid_z);
    checkCudaErrors(cudaMemset(force, 0, n * sizeof(Vec3)));
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    d_compute_forces<<<grid_size_n, block_size_n>>>(position, force, n, box_size, d_result);
    checkCudaErrors(cudaGetLastError());

    double pe;
    checkCudaErrors(cudaMemcpy(&pe, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

__global__ void d_compute_forces_no_pe(const Vec3* position, Vec3* force, unsigned int n, double box_size) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;

    Vec3 f = {0.0, 0.0, 0.0};
    if (i < n && j < n && i != j) {
        const Vec3 pi = position[i];
        const Vec3 pj = position[j];

        // compute distance between particles with periodic boundary conditions
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;

        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        dz -= box_size * nearbyint(dz / box_size);

        // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2 && r2 != 0.0) {
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double fij = -24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            f.x += fij * dx;
            f.y += fij * dy;
            f.z += fij * dz;
        }
    }

    f.x = block_reduce(f.x);
    f.y = block_reduce(f.y);
    f.z = block_reduce(f.z);
    if (threadIdx.x == 0) {
        atomicAdd(&force[i].x, f.x);
        atomicAdd(&force[i].y, f.y);
        atomicAdd(&force[i].z, f.z);
    }
}

void g_compute_forces_no_pe(Vec3* position, Vec3* force, unsigned int n, double box_size) {
    dim3 block_size_n(256, 1);
    unsigned int grid_y = ((n - 1) / block_size_n.y + 1);
    unsigned int grid_y_clamped = (grid_y > 65535U) ? 65535U : grid_y;
    unsigned int grid_z = (grid_y + grid_y_clamped - 1) / grid_y_clamped;
    dim3 grid_size_n((n - 1) / block_size_n.x + 1, grid_y_clamped, grid_z);
    checkCudaErrors(cudaMemset(force, 0, n * sizeof(Vec3)));
    d_compute_forces_no_pe<<<grid_size_n, block_size_n>>>(position, force, n, box_size);
    checkCudaErrors(cudaGetLastError());
}

__global__ void d_second_update(Vec3* velocities, Vec3* forces, unsigned int n, double box_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 vel = velocities[i];
    const Vec3 force = forces[i];
    vel.x += 0.5 * DT * force.x;
    vel.y += 0.5 * DT * force.y;
    vel.z += 0.5 * DT * force.z;
    velocities[i] = vel;
}

inline void g_second_update(Vec3* pos, Vec3* vel, Vec3* force, unsigned int n, double box_size) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    d_second_update<<<grid_size_n, block_size_n>>>(vel, force, n, box_size);
    checkCudaErrors(cudaGetLastError());
}

#ifndef CELL_SKIN
    #define CELL_SKIN 0.3
#endif

typedef struct {
    int nx;
    int ny;
    int nz;
    int n_cells;
    double inv_cx;
    double inv_cy;
    double inv_cz;

    int* d_head;
    int* d_next;
    int* d_pcell;
    double* d_ref_x;
    double* d_ref_y;
    double* d_ref_z;
    int* d_rebuild_flag;

    int* h_head;
    int* h_next;
    int* h_pcell;
    double* h_ref_x;
    double* h_ref_y;
    double* h_ref_z;
    Vec3* h_pos_tmp;
} Tiled3DState;

constexpr double tiled_skin2 = (CELL_SKIN * 0.5) * (CELL_SKIN * 0.5);

__host__ __device__ inline int cell_index_3d(int cx, int cy, int cz, int nx, int ny) {
    return (cz * ny + cy) * nx + cx;
}

void init_tiled3d_state(Tiled3DState* st, unsigned int n, double box_size) {
    memset(st, 0, sizeof(*st));

    double cell_size = R_CUT + CELL_SKIN;
    st->nx = (int)(box_size / cell_size);
    st->ny = (int)(box_size / cell_size);
    st->nz = (int)(box_size / cell_size);
    if (st->nx < 3) st->nx = 3;
    if (st->ny < 3) st->ny = 3;
    if (st->nz < 3) st->nz = 3;
    st->n_cells = st->nx * st->ny * st->nz;
    st->inv_cx = (double)st->nx / box_size;
    st->inv_cy = (double)st->ny / box_size;
    st->inv_cz = (double)st->nz / box_size;

    checkCudaErrors(cudaMalloc((void**)&st->d_head, st->n_cells * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&st->d_next, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&st->d_pcell, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&st->d_ref_x, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&st->d_ref_y, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&st->d_ref_z, n * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&st->d_rebuild_flag, sizeof(int)));

    st->h_head = (int*)malloc(st->n_cells * sizeof(int));
    st->h_next = (int*)malloc(n * sizeof(int));
    st->h_pcell = (int*)malloc(n * sizeof(int));
    st->h_ref_x = (double*)malloc(n * sizeof(double));
    st->h_ref_y = (double*)malloc(n * sizeof(double));
    st->h_ref_z = (double*)malloc(n * sizeof(double));
    st->h_pos_tmp = (Vec3*)malloc(n * sizeof(Vec3));
}

void free_tiled3d_state(Tiled3DState* st) {
    if (st->d_head) checkCudaErrors(cudaFree(st->d_head));
    if (st->d_next) checkCudaErrors(cudaFree(st->d_next));
    if (st->d_pcell) checkCudaErrors(cudaFree(st->d_pcell));
    if (st->d_ref_x) checkCudaErrors(cudaFree(st->d_ref_x));
    if (st->d_ref_y) checkCudaErrors(cudaFree(st->d_ref_y));
    if (st->d_ref_z) checkCudaErrors(cudaFree(st->d_ref_z));
    if (st->d_rebuild_flag) checkCudaErrors(cudaFree(st->d_rebuild_flag));

    free(st->h_head);
    free(st->h_next);
    free(st->h_pcell);
    free(st->h_ref_x);
    free(st->h_ref_y);
    free(st->h_ref_z);
    free(st->h_pos_tmp);

    memset(st, 0, sizeof(*st));
}

__global__ void d_tiled3d_needs_rebuild(const Vec3* position, const double* ref_x, const double* ref_y, const double* ref_z, unsigned int n, double box_size, int* needs_rebuild) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int my_needs_rebuild = 0;
    if (i < (int)n) {
        double dx = position[i].x - ref_x[i];
        double dy = position[i].y - ref_y[i];
        double dz = position[i].z - ref_z[i];
        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        dz -= box_size * nearbyint(dz / box_size);
        if (dx * dx + dy * dy + dz * dz > tiled_skin2) {
            my_needs_rebuild = 1;
        }
    }

    my_needs_rebuild = (int)block_reduce((double)my_needs_rebuild);
    if (threadIdx.x == 0 && my_needs_rebuild) {
        *needs_rebuild = 1;
    }
}

void tiled3d_rebuild(Tiled3DState* st, const Vec3* d_position, unsigned int n) {
    checkCudaErrors(cudaMemcpy(st->h_pos_tmp, d_position, n * sizeof(Vec3), cudaMemcpyDeviceToHost));

    for (int c = 0; c < st->n_cells; c++) {
        st->h_head[c] = -1;
    }

    for (int i = (int)n - 1; i >= 0; i--) {
        int cx = (int)(st->h_pos_tmp[i].x * st->inv_cx);
        int cy = (int)(st->h_pos_tmp[i].y * st->inv_cy);
        int cz = (int)(st->h_pos_tmp[i].z * st->inv_cz);

        if (cx < 0)
            cx = 0;
        else if (cx >= st->nx)
            cx = st->nx - 1;
        if (cy < 0)
            cy = 0;
        else if (cy >= st->ny)
            cy = st->ny - 1;
        if (cz < 0)
            cz = 0;
        else if (cz >= st->nz)
            cz = st->nz - 1;

        int c = cell_index_3d(cx, cy, cz, st->nx, st->ny);
        st->h_pcell[i] = c;
        st->h_next[i] = st->h_head[c];
        st->h_head[c] = i;
        st->h_ref_x[i] = st->h_pos_tmp[i].x;
        st->h_ref_y[i] = st->h_pos_tmp[i].y;
        st->h_ref_z[i] = st->h_pos_tmp[i].z;
    }

    checkCudaErrors(cudaMemcpyAsync(st->d_head, st->h_head, st->n_cells * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(st->d_next, st->h_next, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(st->d_pcell, st->h_pcell, n * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(st->d_ref_x, st->h_ref_x, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(st->d_ref_y, st->h_ref_y, n * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(st->d_ref_z, st->h_ref_z, n * sizeof(double), cudaMemcpyHostToDevice));
}

bool tiled3d_needs_rebuild(Tiled3DState* st, const Vec3* d_position, unsigned int n, double box_size) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    checkCudaErrors(cudaMemset(st->d_rebuild_flag, 0, sizeof(int)));
    d_tiled3d_needs_rebuild<<<grid_size_n, block_size_n>>>(d_position, st->d_ref_x, st->d_ref_y, st->d_ref_z, n, box_size, st->d_rebuild_flag);
    checkCudaErrors(cudaGetLastError());

    int flag = 0;
    checkCudaErrors(cudaMemcpy(&flag, st->d_rebuild_flag, sizeof(int), cudaMemcpyDeviceToHost));
    return flag != 0;
}

__global__ void d_compute_forces_tiled3d(const Vec3* position, Vec3* force, unsigned int n, double box_size, double* result, const int* head, const int* next, const int* pcell, int nx, int ny, int nz) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const Vec3 pi = position[i];
    Vec3 f = {0.0, 0.0, 0.0};
    double pe = 0.0;

    int ci = pcell[i];
    int cix = ci % nx;
    int ciy = (ci / nx) % ny;
    int ciz = ci / (nx * ny);

    for (int ndz = -1; ndz <= 1; ndz++) {
        int ncz = (ciz + ndz + nz) % nz;
        for (int ndy = -1; ndy <= 1; ndy++) {
            int ncy = (ciy + ndy + ny) % ny;
            for (int ndx = -1; ndx <= 1; ndx++) {
                int ncx = (cix + ndx + nx) % nx;
                int nc = cell_index_3d(ncx, ncy, ncz, nx, ny);

                for (int j = head[nc]; j >= 0; j = next[j]) {
                    if ((unsigned int)j == i) {
                        continue;
                    }

                    const Vec3 pj = position[j];
                    double dx = pi.x - pj.x;
                    double dy = pi.y - pj.y;
                    double dz = pi.z - pj.z;

                    dx -= box_size * nearbyint(dx / box_size);
                    dy -= box_size * nearbyint(dy / box_size);
                    dz -= box_size * nearbyint(dz / box_size);

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc2 || r2 == 0.0) {
                        continue;
                    }

                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;

                    double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    f.x += fij * dx;
                    f.y += fij * dy;
                    f.z += fij * dz;

                    pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }
    }

    force[i] = f;
    atomicAdd(result, pe);
}

__global__ void d_compute_forces_no_pe_tiled3d(const Vec3* position, Vec3* force, unsigned int n, double box_size, const int* head, const int* next, const int* pcell, int nx, int ny, int nz) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const Vec3 pi = position[i];
    Vec3 f = {0.0, 0.0, 0.0};

    int ci = pcell[i];
    int cix = ci % nx;
    int ciy = (ci / nx) % ny;
    int ciz = ci / (nx * ny);

    for (int ndz = -1; ndz <= 1; ndz++) {
        int ncz = (ciz + ndz + nz) % nz;
        for (int ndy = -1; ndy <= 1; ndy++) {
            int ncy = (ciy + ndy + ny) % ny;
            for (int ndx = -1; ndx <= 1; ndx++) {
                int ncx = (cix + ndx + nx) % nx;
                int nc = cell_index_3d(ncx, ncy, ncz, nx, ny);

                for (int j = head[nc]; j >= 0; j = next[j]) {
                    if ((unsigned int)j == i) {
                        continue;
                    }

                    const Vec3 pj = position[j];
                    double dx = pi.x - pj.x;
                    double dy = pi.y - pj.y;
                    double dz = pi.z - pj.z;

                    dx -= box_size * nearbyint(dx / box_size);
                    dy -= box_size * nearbyint(dy / box_size);
                    dz -= box_size * nearbyint(dz / box_size);

                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 >= rc2 || r2 == 0.0) {
                        continue;
                    }

                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;

                    double fij = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    f.x += fij * dx;
                    f.y += fij * dy;
                    f.z += fij * dz;
                }
            }
        }
    }

    force[i] = f;
}

inline double g_compute_forces_tiled3d(Vec3* position, Vec3* force, unsigned int n, double box_size, double* d_result, const Tiled3DState* st) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    d_compute_forces_tiled3d<<<grid_size_n, block_size_n>>>(position, force, n, box_size, d_result, st->d_head, st->d_next, st->d_pcell, st->nx, st->ny, st->nz);
    checkCudaErrors(cudaGetLastError());

    double pe;
    checkCudaErrors(cudaMemcpy(&pe, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

inline void g_compute_forces_no_pe_tiled3d(Vec3* position, Vec3* force, unsigned int n, double box_size, const Tiled3DState* st) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    d_compute_forces_no_pe_tiled3d<<<grid_size_n, block_size_n>>>(position, force, n, box_size, st->d_head, st->d_next, st->d_pcell, st->nx, st->ny, st->nz);
    checkCudaErrors(cudaGetLastError());
}

SimulationResult run_simulation_tiled(Particle* particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;

    Vec3* d_position;
    Vec3* d_velocity;
    Vec3* d_force;
    double* d_result;
    Tiled3DState tiled;

    checkCudaErrors(cudaMalloc((void**)&d_position, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));

    init_tiled3d_state(&tiled, n, box_size);

    static_assert(offsetof(Particle, vx) - offsetof(Particle, x) == sizeof(Vec3), "Particle position fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(d_position, sizeof(Vec3), &particles[0].x, sizeof(Particle), sizeof(Vec3), n, cudaMemcpyHostToDevice));
    static_assert(offsetof(Particle, fx) - offsetof(Particle, vx) == sizeof(Vec3), "Particle velocity fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(d_velocity, sizeof(Vec3), &particles[0].vx, sizeof(Particle), sizeof(Vec3), n, cudaMemcpyHostToDevice));

    tiled3d_rebuild(&tiled, d_position, n);

    out.start_potential = g_compute_forces_tiled3d(d_position, d_force, n, box_size, d_result, &tiled);
    out.start_kinetic = g_compute_ke(d_velocity, n, d_result);
    out.start_total = out.start_kinetic + out.start_potential;

#if DEBUG
    double start = omp_get_wtime();
#endif

    for (unsigned int step = 1; step < nsteps; step++) {
        g_first_update(d_position, d_velocity, d_force, n, box_size);
        if (tiled3d_needs_rebuild(&tiled, d_position, n, box_size)) {
            tiled3d_rebuild(&tiled, d_position, n);
        }
        g_compute_forces_no_pe_tiled3d(d_position, d_force, n, box_size, &tiled);
        g_second_update(d_position, d_velocity, d_force, n, box_size);
    }

    g_first_update(d_position, d_velocity, d_force, n, box_size);
    if (tiled3d_needs_rebuild(&tiled, d_position, n, box_size)) {
        tiled3d_rebuild(&tiled, d_position, n);
    }
    out.final_potential = g_compute_forces_tiled3d(d_position, d_force, n, box_size, d_result, &tiled);
    g_second_update(d_position, d_velocity, d_force, n, box_size);
    out.final_kinetic = g_compute_ke(d_velocity, n, d_result);
    out.final_total = out.final_kinetic + out.final_potential;

#if DEBUG
    double stop = omp_get_wtime();
    printf("Time(compute): %f s\n", stop - start);
#endif

    static_assert(offsetof(Particle, vx) - offsetof(Particle, x) == sizeof(Vec3), "Particle position fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].x, sizeof(Particle), d_position, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));
    static_assert(offsetof(Particle, fx) - offsetof(Particle, vx) == sizeof(Vec3), "Particle velocity fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].vx, sizeof(Particle), d_velocity, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));
    static_assert(sizeof(Particle) - offsetof(Particle, fx) == sizeof(Vec3), "Particle force fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].fx, sizeof(Particle), d_force, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));

    out.n = n;
    out.particles = particles;
    return out;
}

int initialize_particles(Particle* particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    srand(seed);
    unsigned int n_side = (unsigned int)ceil(cbrt((double)n));
    double placement_size = placement_fraction * box_size;
    double offset = 0.5 * (box_size - placement_size);
    double delta = placement_size / (double)n_side;

    double mean_vx = 0.0;
    double mean_vy = 0.0;
    double mean_vz = 0.0;
    // place particles int he middle of the grid with some random jitter and assign random velocities
    for (unsigned int k = 0; k < n; k++) {
        particles[k].id = k;
        unsigned int ix = k % n_side;
        unsigned int iy = (k / n_side) % n_side;
        unsigned int iz = k / (n_side * n_side);

        double x0 = offset + (0.5 + (double)ix) * delta;
        double y0 = offset + (0.5 + (double)iy) * delta;
        double z0 = offset + (0.5 + (double)iz) * delta;

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].z = z0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;
        particles[k].vz = 2.0 * random_double() - 1.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
        mean_vz += particles[k].vz;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    mean_vz /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        particles[k].vz -= mean_vz;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy + particles[k].vz * particles[k].vz);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx *= scale;
        particles[k].vy *= scale;
        particles[k].vz *= scale;
    }

    return 1;
}

// apply periodic boundary conditions to ensure particles stay within the simulation box
void wrap_positions(Particle* particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        Particle* p = &particles[i];
        double wx = fmod(p->x, box_size);
        double wy = fmod(p->y, box_size);
        double wz = fmod(p->z, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }
        if (wz < 0.0) {
            wz += box_size;
        }

        p->x = wx;
        p->y = wy;
        p->z = wz;
    }
}

double compute_forces(Particle* particles, unsigned int n, double box_size) {
    for (unsigned int i = 0; i < n; ++i) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }
    double pe = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            Particle* pi = &particles[i];
            Particle* pj = &particles[j];

            // compute distance between particles with periodic boundary conditions
            double dx = pj->x - pi->x;
            double dy = pj->y - pi->y;
            double dz = pj->z - pi->z;

            dx -= box_size * nearbyint(dx / box_size);
            dy -= box_size * nearbyint(dy / box_size);
            dz -= box_size * nearbyint(dz / box_size);

            // compute Lennard-Jones force and potential energy contribution if particles are within the cutoff distance
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r >= R_CUT || r == 0.0) {
                continue;
            }
            double sr = SIGMA / r;

            double fij = -24.0 * EPSILON * (2.0 * pow(sr, 12.0) - pow(sr, 6.0)) / r;
            double fx = fij * dx / r;
            double fy = fij * dy / r;
            double fz = fij * dz / r;

            pi->fx += fx;
            pi->fy += fy;
            pi->fz += fz;

            double vij = 4.0 * EPSILON * (pow(sr, 12.0) - pow(sr, 6.0)) - v_shift;
            pe += 0.5 * vij;
        }
    }

    return pe;
}

double leapfrog_step(Particle* particles, unsigned int n, double box_size) {
    // update velocities by half a time step, then update positions by a full time step,
    //and finally update velocities by another half time step to complete the leapfrog integration step
    for (unsigned int i = 0; i < n; ++i) {
        Particle* p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->vz += 0.5 * DT * p->fz;

        p->x += DT * p->vx;
        p->y += DT * p->vy;
        p->z += DT * p->vz;
    }

    wrap_positions(particles, n, box_size);

    double pe = compute_forces(particles, n, box_size);

    for (unsigned int i = 0; i < n; ++i) {
        Particle* p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
        p->vz += 0.5 * DT * p->fz;
    }

    return pe;
}

SimulationResult run_simulation(Particle* particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    // assume log_steps = 0
    SimulationResult out;

    Vec3* d_position;
    Vec3* d_velocity;
    Vec3* d_force;
    double* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_position, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));

    // abuse Memcpy2D to do strided copies
    static_assert(offsetof(Particle, vx) - offsetof(Particle, x) == sizeof(Vec3), "Particle position fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(d_position, sizeof(Vec3), &particles[0].x, sizeof(Particle), sizeof(Vec3), n, cudaMemcpyHostToDevice));
    static_assert(offsetof(Particle, fx) - offsetof(Particle, vx) == sizeof(Vec3), "Particle velocity fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(d_velocity, sizeof(Vec3), &particles[0].vx, sizeof(Particle), sizeof(Vec3), n, cudaMemcpyHostToDevice));
    // no need to copy forces as they will be zeroed/computed down bellow:

    out.start_potential = g_compute_forces(d_position, d_force, n, box_size, d_result);
    out.start_kinetic = g_compute_ke(d_velocity, n, d_result);
    out.start_total = out.start_kinetic + out.start_potential;

#if DEBUG
    double start = omp_get_wtime();
#endif

    // do nsteps-1 steps without energy computation
    for (unsigned int step = 1; step < nsteps; step++) {
        g_first_update(d_position, d_velocity, d_force, n, box_size);
        g_compute_forces_no_pe(d_position, d_force, n, box_size);
        g_second_update(d_position, d_velocity, d_force, n, box_size);
    }

    // do last step with energy computation
    g_first_update(d_position, d_velocity, d_force, n, box_size);
    out.final_potential = g_compute_forces(d_position, d_force, n, box_size, d_result);
    g_second_update(d_position, d_velocity, d_force, n, box_size);
    out.final_kinetic = g_compute_ke(d_velocity, n, d_result);
    out.final_total = out.final_kinetic + out.final_potential;

#if DEBUG
    double stop = omp_get_wtime();
    printf("Time(compute): %f s\n", stop - start);
#endif

    // abuse Memcpy2D for strided copies
    static_assert(offsetof(Particle, vx) - offsetof(Particle, x) == sizeof(Vec3), "Particle position fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].x, sizeof(Particle), d_position, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));
    static_assert(offsetof(Particle, fx) - offsetof(Particle, vx) == sizeof(Vec3), "Particle velocity fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].vx, sizeof(Particle), d_velocity, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));
    static_assert(sizeof(Particle) - offsetof(Particle, fx) == sizeof(Vec3), "Particle force fields are not layout-compatible with double3");
    checkCudaErrors(cudaMemcpy2D(&particles[0].fx, sizeof(Particle), d_force, sizeof(Vec3), sizeof(Vec3), n, cudaMemcpyDeviceToHost));

    out.n = n;
    out.particles = particles;
    return out;
}