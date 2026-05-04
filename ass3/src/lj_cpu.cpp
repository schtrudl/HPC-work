#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>

#include "gifenc.h"
#include "lennard-jones.h"

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
        int px = (int)(particles[i].x / box_size * (double)(FRAME_WIDTH - 1));
        int py = (int)(particles[i].y / box_size * (double)(FRAME_HEIGHT - 1));
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
double inline compute_ke(const Particle* particles, unsigned int n) {
    double ke = 0.0;
    OMP(parallel for reduction(+:ke))
    for (unsigned int i = 0; i < n; ++i) {
        const Particle* p = &particles[i];
        ke += 0.5 * (p->vx * p->vx + p->vy * p->vy);
    }
    return ke;
}

int initialize_particles(Particle* particles, unsigned int n, double box_size, double placement_fraction, unsigned int seed, double temperature) {
    srand(seed);
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

        particles[k].x = x0 + (2.0 * random_double() - 1.0) * JITTER * delta;
        particles[k].y = y0 + (2.0 * random_double() - 1.0) * JITTER * delta;

        particles[k].vx = 2.0 * random_double() - 1.0;
        particles[k].vy = 2.0 * random_double() - 1.0;

        mean_vx += particles[k].vx;
        mean_vy += particles[k].vy;
    }

    mean_vx /= (double)n;
    mean_vy /= (double)n;
    double ke = 0.0;
    // subtract mean velocity to ensure zero net momentum and compute initial kinetic energy
    for (unsigned int k = 0; k < n; k++) {
        particles[k].vx -= mean_vx;
        particles[k].vy -= mean_vy;
        ke += 0.5 * (particles[k].vx * particles[k].vx + particles[k].vy * particles[k].vy);
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
    }

    return 1;
}

void inline first_update(Particle* particles, unsigned int n, double box_size) {
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        Particle* p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;

        // fused wrap_positions
        // apply periodic boundary conditions to ensure particles stay within the simulation box
        double wx = fmod(p->x + DT * p->vx, box_size);
        double wy = fmod(p->y + DT * p->vy, box_size);

        if (wx < 0.0) {
            wx += box_size;
        }
        if (wy < 0.0) {
            wy += box_size;
        }

        p->x = wx;
        p->y = wy;
    }
}

// ============================================================
// Cell list with Verlet skin: cells of size (R_CUT + CELL_SKIN).
// The 3x3 neighbourhood is guaranteed to contain all pairs within
// R_CUT as long as no particle has moved more than CELL_SKIN/2
// from its position at the last rebuild.
// ============================================================

#define CELL_SKIN 0.3 /* extra buffer beyond R_CUT; tune for performance */

typedef struct {
    int nx, ny, n_cells;
    double inv_cx, inv_cy; /* 1/cell_size for fast index computation */
    double box_size; /* needed for minimum-image displacement check */
    double skin2; /* (CELL_SKIN/2)^2 — rebuild trigger threshold  */
    int* head; /* [n_cells] linked-list head per cell, -1 = empty */
    int* next; /* [n]       next particle in same cell, -1 = end  */
    int* pcell; /* [n]       current cell index per particle        */
    double* ref_x; /* [n]       particle x at last rebuild             */
    double* ref_y; /* [n]       particle y at last rebuild             */
} CellList;

static inline int cl_cell(const CellList* cl, double x, double y) {
    int cx = (int)(x * cl->inv_cx);
    int cy = (int)(y * cl->inv_cy);
    if (cx < 0)
        cx = 0;
    else if (cx >= cl->nx)
        cx = cl->nx - 1;
    if (cy < 0)
        cy = 0;
    else if (cy >= cl->ny)
        cy = cl->ny - 1;
    return cy * cl->nx + cx;
}

static CellList* cl_create(unsigned int n, double box_size) {
    CellList* cl = (CellList*)malloc(sizeof(CellList));
    if (!cl) return NULL;
    /* cell_size = R_CUT + CELL_SKIN ensures the 3x3 neighbourhood covers all
       pairs within R_CUT even if the list is slightly stale (up to skin/2
       displacement per particle).  Enforce nx >= 3 so periodic wrapping of
       the neighbourhood is always valid. */
    double cell_size = R_CUT + CELL_SKIN;
    cl->nx = (int)(box_size / cell_size);
    cl->ny = (int)(box_size / cell_size);
    if (cl->nx < 3) cl->nx = 3;
    if (cl->ny < 3) cl->ny = 3;
    cl->n_cells = cl->nx * cl->ny;
    cl->inv_cx = (double)cl->nx / box_size;
    cl->inv_cy = (double)cl->ny / box_size;
    cl->box_size = box_size;
    cl->skin2 = (CELL_SKIN * 0.5) * (CELL_SKIN * 0.5);
    cl->head = (int*)malloc(cl->n_cells * sizeof(int));
    cl->next = (int*)malloc(n * sizeof(int));
    cl->pcell = (int*)malloc(n * sizeof(int));
    cl->ref_x = (double*)malloc(n * sizeof(double));
    cl->ref_y = (double*)malloc(n * sizeof(double));
    return cl;
}

/* Reset and refill in-place — O(n), no malloc.
   Also snapshots current positions as new rebuild reference. */
static void cl_rebuild(CellList* cl, const Particle* particles, unsigned int n) {
    for (int c = 0; c < cl->n_cells; c++)
        cl->head[c] = -1;
    /* Iterate backwards so the linked list preserves ascending particle order. */
    for (int i = (int)n - 1; i >= 0; i--) {
        int c = cl_cell(cl, particles[i].x, particles[i].y);
        cl->pcell[i] = c;
        cl->next[i] = cl->head[c];
        cl->head[c] = i;
        cl->ref_x[i] = particles[i].x;
        cl->ref_y[i] = particles[i].y;
    }
}

/* Returns 1 if any particle has moved more than CELL_SKIN/2 from its
   reference position (using minimum-image convention to handle wraps). */
static int cl_needs_rebuild(const CellList* cl, const Particle* particles, unsigned int n) {
    const double bs = cl->box_size;
    const double skin2 = cl->skin2;
    for (unsigned int i = 0; i < n; i++) {
        double dx = particles[i].x - cl->ref_x[i];
        double dy = particles[i].y - cl->ref_y[i];
        /* minimum-image: avoids false positives when a particle wraps */
        dx -= bs * nearbyint(dx / bs);
        dy -= bs * nearbyint(dy / bs);
        if (dx * dx + dy * dy > skin2) return 1;
    }
    return 0;
}

// shift potential to ensure it goes to zero at the cutoff distance, improving energy conservation
const double v_shift = 4.0 * EPSILON * (pow(SIGMA / R_CUT, 12.0) - pow(SIGMA / R_CUT, 6.0));

// ---- Tiled force computation using the cell list ----

double inline compute_forces(Particle* particles, unsigned int n, double box_size, const CellList* cl) {
    const double rc2 = R_CUT * R_CUT;
    double pe = 0.0;
    OMP(parallel for reduction(+:pe))
    for (unsigned int i = 0; i < n; ++i) {
        Particle* pi = &particles[i];
        double fix = 0.0, fiy = 0.0;
        int ci = cl->pcell[i];
        int cix = ci % cl->nx;
        int ciy = ci / cl->nx;
        for (int ndy = -1; ndy <= 1; ndy++) {
            int ncy = (ciy + ndy + cl->ny) % cl->ny;
            for (int ndx = -1; ndx <= 1; ndx++) {
                int ncx = (cix + ndx + cl->nx) % cl->nx;
                int nc = ncy * cl->nx + ncx;
                for (int j = cl->head[nc]; j >= 0; j = cl->next[j]) {
                    if (j == (int)i) continue;
                    const Particle* pj = &particles[j];
                    double dx = pi->x - pj->x;
                    double dy = pi->y - pj->y;
                    dx -= box_size * nearbyint(dx / box_size);
                    dy -= box_size * nearbyint(dy / box_size);
                    double r2 = dx * dx + dy * dy;
                    if (r2 >= rc2 || r2 == 0.0) continue;
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij_r2 * dx;
                    fiy += fij_r2 * dy;
                    pe += 0.5 * (4.0 * EPSILON * (sr12 - sr6) - v_shift);
                }
            }
        }
        pi->fx = fix;
        pi->fy = fiy;
    }
    return pe;
}

void inline compute_forces_no_pe(Particle* particles, unsigned int n, double box_size, const CellList* cl) {
    const double rc2 = R_CUT * R_CUT;
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        Particle* pi = &particles[i];
        double fix = 0.0, fiy = 0.0;
        int ci = cl->pcell[i];
        int cix = ci % cl->nx;
        int ciy = ci / cl->nx;
        for (int ndy = -1; ndy <= 1; ndy++) {
            int ncy = (ciy + ndy + cl->ny) % cl->ny;
            for (int ndx = -1; ndx <= 1; ndx++) {
                int ncx = (cix + ndx + cl->nx) % cl->nx;
                int nc = ncy * cl->nx + ncx;
                for (int j = cl->head[nc]; j >= 0; j = cl->next[j]) {
                    if (j == (int)i) continue;
                    const Particle* pj = &particles[j];
                    double dx = pi->x - pj->x;
                    double dy = pi->y - pj->y;
                    dx -= box_size * nearbyint(dx / box_size);
                    dy -= box_size * nearbyint(dy / box_size);
                    double r2 = dx * dx + dy * dy;
                    if (r2 >= rc2 || r2 == 0.0) continue;
                    double sr2 = (SIGMA * SIGMA) / r2;
                    double sr6 = sr2 * sr2 * sr2;
                    double sr12 = sr6 * sr6;
                    double fij_r2 = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
                    fix += fij_r2 * dx;
                    fiy += fij_r2 * dy;
                }
            }
        }
        pi->fx = fix;
        pi->fy = fiy;
    }
}

void inline second_update(Particle* particles, unsigned int n) {
    OMP(parallel for)
    for (unsigned int i = 0; i < n; ++i) {
        Particle* p = &particles[i];
        p->vx += 0.5 * DT * p->fx;
        p->vy += 0.5 * DT * p->fy;
    }
}

SimulationResult run_simulation(Particle* particles, unsigned int n, unsigned int nsteps, double box_size, int log_steps) {
    SimulationResult out;

    // XXX: we could move this to init but that would feel like cheating
    CellList* cl = cl_create(n, box_size);
    cl_rebuild(cl, particles, n);

    // XXX: we could move this to init too
    out.start_potential = compute_forces(particles, n, box_size, cl);
    out.start_kinetic = compute_ke(particles, n);
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
        first_update(particles, n, box_size);
        // Rebuild cell list only when a particle has moved > CELL_SKIN/2
        if (cl_needs_rebuild(cl, particles, n)) {
            cl_rebuild(cl, particles, n);
        }
        compute_forces_no_pe(particles, n, box_size, cl);
        second_update(particles, n);
#if GENERATE_GIF
        if (gif && FRAME_EVERY > 0 && (step + 1) % FRAME_EVERY == 0) {
            render_frame_gif(gif, particles, n, box_size);
            ge_add_frame(gif, FRAME_DELAY);
        }
#endif
    }

    for (unsigned int step = steps_without_log; step < nsteps; step++) {
        first_update(particles, n, box_size);
        if (cl_needs_rebuild(cl, particles, n)) {
            cl_rebuild(cl, particles, n);
        }
        out.final_potential = compute_forces(particles, n, box_size, cl);
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

#if GENERATE_GIF
    if (gif) {
        ge_close_gif(gif);
    }
#endif

    out.n = n;
    out.particles = particles;
    return out;
    // yes we leak, but we do not care
}

#include "main.h"