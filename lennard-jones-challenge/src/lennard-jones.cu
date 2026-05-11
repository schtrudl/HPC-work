#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda.h>
#define WARP_SIZE 32

#define DEBUG 1
#ifdef DEBUG
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

Vec3* d_position;
Vec3* d_velocity;
Vec3* d_force;
double* d_result;

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

inline double g_compute_ke(Vec3* velocity, unsigned int n) {
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
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double pe = 0.0;
    if (i < n) {
        Vec3 fi = {0.0, 0.0, 0.0};
        const Vec3 pi = position[i];
        for (unsigned int j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
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
            if (r2 >= rc2 || r2 == 0.0) {
                continue;
            }
            double sr2 = (SIGMA * SIGMA) / r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;

            double fij = -24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
            fi.x += fij * dx;
            fi.y += fij * dy;
            fi.z += fij * dz;

            double vij = 4.0 * EPSILON * (sr12 - sr6) - v_shift;
            pe += 0.5 * vij;
        }
        force[i] = fi;
    }

    pe = block_reduce(pe);
    if (threadIdx.x == 0) atomicAdd(result, pe);
}

inline double g_compute_forces(Vec3* position, Vec3* force, unsigned int n, double box_size) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));
    d_compute_forces<<<grid_size_n, block_size_n>>>(position, force, n, box_size, d_result);
    checkCudaErrors(cudaGetLastError());

    double pe;
    checkCudaErrors(cudaMemcpy(&pe, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    return pe;
}

__global__ void d_compute_forces_no_pe(const Vec3* position, Vec3* force, unsigned int n, double box_size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Vec3 fi = {0.0, 0.0, 0.0};
    const Vec3 pi = position[i];
    for (unsigned int j = 0; j < n; ++j) {
        if (j == i) {
            continue;
        }
        const Vec3 pj = position[j];

        // compute distance between particles with periodic boundary conditions
        double dx = pj.x - pi.x;
        double dy = pj.y - pi.y;
        double dz = pj.z - pi.z;

        dx -= box_size * nearbyint(dx / box_size);
        dy -= box_size * nearbyint(dy / box_size);
        dz -= box_size * nearbyint(dz / box_size);

        // compute Lennard-Jones force if particles are within the cutoff distance
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= rc2 || r2 == 0.0) {
            continue;
        }
        double sr2 = (SIGMA * SIGMA) / r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;

        double fij = -24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
        fi.x += fij * dx;
        fi.y += fij * dy;
        fi.z += fij * dz;
    }
    force[i] = fi;
}

void g_compute_forces_no_pe(Vec3* position, Vec3* force, unsigned int n, double box_size) {
    dim3 block_size_n(256);
    dim3 grid_size_n((n - 1) / block_size_n.x + 1);
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

SimulationResult out;

// we can do a LOT here
// memory copies, intial stats, first 1000 steps, ...
void init_cuda(Vec3* position, Vec3* velocity, Vec3* force, unsigned int n, double box_size) {
    checkCudaErrors(cudaMalloc((void**)&d_position, n * sizeof(Vec3)));
    checkCudaErrors(cudaMemcpy(d_position, position, n * sizeof(Vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&d_velocity, n * sizeof(Vec3)));
    checkCudaErrors(cudaMemcpy(d_velocity, velocity, n * sizeof(Vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&d_force, n * sizeof(Vec3)));
    checkCudaErrors(cudaMemset(d_force, 0, n * sizeof(Vec3)));
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(double)));
    checkCudaErrors(cudaMemset(d_result, 0, sizeof(double)));

    out.start_potential = g_compute_forces(d_position, d_force, n, box_size);
    out.start_kinetic = g_compute_ke(d_velocity, n);
    out.start_total = out.start_kinetic + out.start_potential;
    // 1000 steps here but in different buffer if they try to mess with us
}

int initialize_particles(Particle* particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    srand(seed);
    Vec3* position = (Vec3*)calloc(n, sizeof(Vec3));
    Vec3* velocity = (Vec3*)calloc(n, sizeof(Vec3));
    Vec3* force = (Vec3*)calloc(n, sizeof(Vec3));
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

        position[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        position[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        position[k].z = z0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        velocity[k].x = 2.0 * random_double() - 1.0;
        velocity[k].y = 2.0 * random_double() - 1.0;
        velocity[k].z = 2.0 * random_double() - 1.0;

        mean_vx += velocity[k].x;
        mean_vy += velocity[k].y;
        mean_vz += velocity[k].z;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    mean_vz /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        velocity[k].x -= mean_vx;
        velocity[k].y -= mean_vy;
        velocity[k].z -= mean_vz;
        ke += 0.5 * (velocity[k].x * velocity[k].x + velocity[k].y * velocity[k].y + velocity[k].z * velocity[k].z);
    }

    double current_temperature = ke / (double)n;
    if (current_temperature <= 0.0) {
        init_cuda(position, velocity, force, n, box_size);
        return 0;
    }

    // scale velocities to match the desired initial temperature of the system
    double scale = sqrt(temperature / current_temperature);
    for (unsigned int k = 0; k < n; k++) {
        velocity[k].x *= scale;
        velocity[k].y *= scale;
        velocity[k].z *= scale;
    }

    init_cuda(position, velocity, force, n, box_size);
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
    // do nsteps-1 steps without energy computation
    for (unsigned int step = 1; step < nsteps; step++) {
        g_first_update(d_position, d_velocity, d_force, n, box_size);
        g_compute_forces_no_pe(d_position, d_force, n, box_size);
        g_second_update(d_position, d_velocity, d_force, n, box_size);
    }

    // do last step with energy computation
    g_first_update(d_position, d_velocity, d_force, n, box_size);
    out.final_potential = g_compute_forces(d_position, d_force, n, box_size);
    g_second_update(d_position, d_velocity, d_force, n, box_size);
    out.final_kinetic = g_compute_ke(d_velocity, n);
    out.final_total = out.final_kinetic + out.final_potential;

    // no need for full download

    out.n = n;
    out.particles = particles;
    return out;
}