#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

// TODO(perf): inlining

// reporting of timings of substeps (useful for fine-tuning)
// beware that full timings are in this mode not representative
#define REPORT_SUB_TIMES 0
// koliko seamov bomo obdelali na enkrat
//
// po definiciji problema naj bi bila kle 1
#define SEAMS_PER_ROUND 1
#define PARALLEL 1

#if REPORT_SUB_TIMES
    #define report_time_into_var(var, name, code) \
        do { \
            double start = omp_get_wtime(); \
            var = code; \
            double stop = omp_get_wtime(); \
            printf("Time(%s): %f s\n", name, stop - start); \
        } while (0)
    #define report_time(name, code) \
        do { \
            double start = omp_get_wtime(); \
            code; \
            double stop = omp_get_wtime(); \
            printf("Time(%s): %f s\n", name, stop - start); \
        } while (0)
#else
    #define report_time(name, code) code
    #define report_time_into_var(var, name, code) var = code
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 3
#define MAX_FILENAME 255
#if PARALLEL
    // this is to prevent formating of the OMP pragmas
    #define OMP(x) DO_PRAGMA(omp x)
#else
    #define OMP(x)
#endif

#define DO_PRAGMA(x) _Pragma(#x)

// typed malloc
#define box(n, type) (type*)aligned_alloc(sizeof(type), (n) * sizeof(type))
// type aliases for better readability
#define usize size_t
#define u8 uint8_t
#define u32 uint32_t
#define i32 int32_t
#define f64 double
#define f32 float

// some floating typing alias
#define fxx f32
// buffer type alias
#define txx u32

// this struct makes it easier to work with the image data
// as c indexing automatically takes into account sizeof(Pixel)
typedef struct {
    u8 r;
    u8 g;
    u8 b;
} Pixel;

inline usize min(const usize a, const usize b) {
    return (a < b) ? a : b;
}

inline txx min_txx(const txx a, const txx b) {
    return (a < b) ? a : b;
}

inline txx min_txx_3(const txx a, const txx b, const txx c) {
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

inline usize min_col(const usize a, const usize b, const txx a_val, const txx b_val) {
    return (a_val < b_val) ? a : b;
}

inline usize min_col_3(const usize a, const usize b, const usize c, const txx a_val, const txx b_val, const txx c_val) {
    return (a_val < b_val) ? ((a_val < c_val) ? a : c) : ((b_val < c_val) ? b : c);
}

void compute_energy(const Pixel* const image, const usize width, const usize stride, const usize height, txx* const energy) {
    OMP(parallel for)  //
    for (usize row = 0; row < height; row++) {
        const usize row_minus_1 = (row == 0) ? 0 : row - 1;
        const usize row_plus_1 = (row == height - 1) ? row : row + 1;
        // precompute offsets
        const Pixel* const row_top = image + row_minus_1 * stride;
        const Pixel* const row_mid = image + row * stride;
        const Pixel* const row_bot = image + row_plus_1 * stride;
        txx* const energy_row = energy + row * width;

        // edge cases are problematic for SIMD
        // col = 0
        {
            const usize col = 0;
            const Pixel tl = row_top[0], tc = row_top[0], tr = row_top[1];
            const Pixel ml = row_mid[0], mr = row_mid[1];
            const Pixel bl = row_bot[0], bc = row_bot[0], br = row_bot[1];

            const f32 Gxr = -(f32)tl.r - 2.0f * (f32)ml.r - (f32)bl.r + (f32)tr.r + 2.0f * (f32)mr.r + (f32)br.r;
            const f32 Gxg = -(f32)tl.g - 2.0f * (f32)ml.g - (f32)bl.g + (f32)tr.g + 2.0f * (f32)mr.g + (f32)br.g;
            const f32 Gxb = -(f32)tl.b - 2.0f * (f32)ml.b - (f32)bl.b + (f32)tr.b + 2.0f * (f32)mr.b + (f32)br.b;

            const f32 Gyr = (f32)tl.r + 2.0f * (f32)tc.r + (f32)tr.r - (f32)bl.r - 2.0f * (f32)bc.r - (f32)br.r;
            const f32 Gyg = (f32)tl.g + 2.0f * (f32)tc.g + (f32)tr.g - (f32)bl.g - 2.0f * (f32)bc.g - (f32)br.g;
            const f32 Gyb = (f32)tl.b + 2.0f * (f32)tc.b + (f32)tr.b - (f32)bl.b - 2.0f * (f32)bc.b - (f32)br.b;

            const f32 mag = (sqrtf(Gxr * Gxr + Gyr * Gyr) + sqrtf(Gxg * Gxg + Gyg * Gyg) + sqrtf(Gxb * Gxb + Gyb * Gyb)) * (1.0f / 3.0f);
            energy_row[col] = (txx)mag;
        }

        // process inner cols with SIMD
        OMP(simd)
        for (usize col = 1; col < width - 1; col++) {
            // load all, stack is aligned, so this should make it obvious for compiler to SIMDfy as Vec3A
            const Pixel tl = row_top[col - 1], tc = row_top[col], tr = row_top[col + 1];
            const Pixel ml = row_mid[col - 1], mr = row_mid[col + 1];
            const Pixel bl = row_bot[col - 1], bc = row_bot[col], br = row_bot[col + 1];

            // hopefully compute with SIMD as component-wise vector ops
            const f32 Gxr = -(f32)tl.r - 2.0f * (f32)ml.r - (f32)bl.r + (f32)tr.r + 2.0f * (f32)mr.r + (f32)br.r;
            const f32 Gxg = -(f32)tl.g - 2.0f * (f32)ml.g - (f32)bl.g + (f32)tr.g + 2.0f * (f32)mr.g + (f32)br.g;
            const f32 Gxb = -(f32)tl.b - 2.0f * (f32)ml.b - (f32)bl.b + (f32)tr.b + 2.0f * (f32)mr.b + (f32)br.b;

            const f32 Gyr = (f32)tl.r + 2.0f * (f32)tc.r + (f32)tr.r - (f32)bl.r - 2.0f * (f32)bc.r - (f32)br.r;
            const f32 Gyg = (f32)tl.g + 2.0f * (f32)tc.g + (f32)tr.g - (f32)bl.g - 2.0f * (f32)bc.g - (f32)br.g;
            const f32 Gyb = (f32)tl.b + 2.0f * (f32)tc.b + (f32)tr.b - (f32)bl.b - 2.0f * (f32)bc.b - (f32)br.b;

            // and pray that it works
            const f32 mag = (sqrtf(Gxr * Gxr + Gyr * Gyr) + sqrtf(Gxg * Gxg + Gyg * Gyg) + sqrtf(Gxb * Gxb + Gyb * Gyb)) * (1.0f / 3.0f);
            energy_row[col] = (txx)mag;
        }

        // col = width - 1
        {
            const usize col = width - 1;
            const Pixel tl = row_top[col - 1], tc = row_top[col], tr = row_top[col];
            const Pixel ml = row_mid[col - 1], mr = row_mid[col];
            const Pixel bl = row_bot[col - 1], bc = row_bot[col], br = row_bot[col];

            const f32 Gxr = -(f32)tl.r - 2.0f * (f32)ml.r - (f32)bl.r + (f32)tr.r + 2.0f * (f32)mr.r + (f32)br.r;
            const f32 Gxg = -(f32)tl.g - 2.0f * (f32)ml.g - (f32)bl.g + (f32)tr.g + 2.0f * (f32)mr.g + (f32)br.g;
            const f32 Gxb = -(f32)tl.b - 2.0f * (f32)ml.b - (f32)bl.b + (f32)tr.b + 2.0f * (f32)mr.b + (f32)br.b;

            const f32 Gyr = (f32)tl.r + 2.0f * (f32)tc.r + (f32)tr.r - (f32)bl.r - 2.0f * (f32)bc.r - (f32)br.r;
            const f32 Gyg = (f32)tl.g + 2.0f * (f32)tc.g + (f32)tr.g - (f32)bl.g - 2.0f * (f32)bc.g - (f32)br.g;
            const f32 Gyb = (f32)tl.b + 2.0f * (f32)tc.b + (f32)tr.b - (f32)bl.b - 2.0f * (f32)bc.b - (f32)br.b;

            const f32 mag = (sqrtf(Gxr * Gxr + Gyr * Gyr) + sqrtf(Gxg * Gxg + Gyg * Gyg) + sqrtf(Gxb * Gxb + Gyb * Gyb)) * (1.0f / 3.0f);
            energy_row[col] = (txx)mag;
        }
    }
}

void compute_cumulative(const txx* const energy, const usize width, const usize height, txx* const cumulative) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    // TODO(perf): implement parallel with dependency triangles
    const usize height_minus_1 = height - 1;

    OMP(parallel)  //
    {
        // bottom up
        for (usize row = height_minus_1; row < height; row--) {  // this is so wrong that it actually works
            OMP(for) //
            for (usize col = 0; col < width; col++) {
                const usize idx = (row * width + col);
                if (row == height_minus_1) {
                    cumulative[idx] = energy[idx];
                    continue;
                }
                const usize idx_prev = ((row + 1) * width + col);

                if (col == 0) {
                    cumulative[idx] = energy[idx] + min_txx(cumulative[idx_prev], cumulative[idx_prev + 1]);
                } else if (col == width - 1) {
                    cumulative[idx] = energy[idx] + min_txx(cumulative[idx_prev - 1], cumulative[idx_prev]);
                } else {
                    cumulative[idx] = energy[idx] + min_txx_3(cumulative[idx_prev - 1], cumulative[idx_prev], cumulative[idx_prev + 1]);
                }
            }
        }
    }
}

// sprehodi po commulativi in poišče stolpce z najmanjšimi energijami
// shrani jih v seam (height * n_seams_to_remove) in vrne kok seamov je naredil
// lahko jih je manj bo pa vsaj en
size_t find_seam(const txx* const cumulative, const usize width, const usize height, const usize n_seams_to_remove, usize* const seam) {
    // TODO(perf): implement for n > 1 seams
    txx minimum = cumulative[0];
    usize min_column = 0;

    // top-down
    for (usize col = 0; col < width; col++) {
        if (cumulative[col] < minimum) {
            minimum = cumulative[col];
            min_column = col;
        }
    }
    seam[0] = min_column;

    for (usize row = 1; row < height; row++) {
        const usize prev_col = seam[row - 1];
        if (prev_col == 0) {
            seam[row] = min_col(0, 1, cumulative[row * width + 0], cumulative[row * width + 1]);
        } else if (prev_col == width - 1) {
            seam[row] = min_col(width - 2, width - 1, cumulative[row * width + width - 2], cumulative[row * width + width - 1]);
        } else {
            seam[row] = min_col_3(prev_col - 1, prev_col, prev_col + 1, cumulative[row * width + prev_col - 1], cumulative[row * width + prev_col], cumulative[row * width + prev_col + 1]);
        }
    }

    return 1;
}

void remove_seam(Pixel* const image, const usize width, const usize stride, const usize height, const usize n_seams, const usize* const seam) {
    if (n_seams == 0) return;
    // odstrani stolpce podane v seams iz slike in shrani v image_out
    OMP(parallel for) // TODO(perf): paralization probably not worth for small images
    for (usize row = 0; row < height; row++) {
        usize s = 1;
        // assumes ordered seams
        for (usize col = seam[row * n_seams + 0] + 1; col < width; col++) {
            if (s < n_seams && col == seam[row * n_seams + s]) {
                s++;
                continue;
            }
            const usize idx = (row * stride + col);
            image[idx - s] = image[idx];
        }
    }
}

void remove_1seam(Pixel* const image, const usize width, const usize stride, const usize height, const usize* const seam) {
    // odstrani stolpce podane v seams iz slike in shrani v image_out
    OMP(parallel for) // TODO(perf): paralization probably not worth for small images
    for (usize row = 0; row < height; row++) {
        const usize s = seam[row];
        for (usize col = s + 1; col < width; col++) {
            const usize idx = (row * stride + col);
            image[idx - 1] = image[idx];  // TODO(perf): memcpy?
        }
    }
}

Pixel* main_algo(Pixel* image, usize* width, const usize height, const usize width_to_remove, txx* const energy_buffer, txx* const cumulative, usize* const seam) {
    const usize stride = *width;
    usize n_seams_to_remove = width_to_remove;
    while (n_seams_to_remove > 0) {
        report_time("compute_energy", compute_energy(image, *width, stride, height, energy_buffer));
        report_time("compute_cumulative", compute_cumulative(energy_buffer, *width, height, cumulative));
        usize s;
        report_time_into_var(s, "find_seam", find_seam(cumulative, *width, height, min(SEAMS_PER_ROUND, n_seams_to_remove), seam));
        //report_time("remove_seam", remove_seam(image, *width, stride, height, s, seam));
        report_time("remove_1seam", remove_1seam(image, *width, stride, height, seam));
        n_seams_to_remove -= s;
        *width -= s;
    }
    return image;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("USAGE: sample input_image width_to_remove output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[3]);

    usize width_to_remove = (usize)atoi(argv[2]);

    // Load image from file and allocate space for the output image
    int orig_width, orig_height, cpp;
    Pixel* image_in = (Pixel*)stbi_load(image_in_name, &orig_width, &orig_height, &cpp, 3);

    if (image_in == NULL) {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    usize width = (usize)orig_width;
    usize height = (usize)orig_height;
    printf("Loaded image %s of size %zux%zu.\n", image_in_name, width, height);

    txx* energy_buffer = box(width * height, txx);
    txx* cumulative = box(width * height, txx);
    usize* seam = box(height * SEAMS_PER_ROUND, usize);

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    Pixel* im = main_algo(image_in, &width, height, width_to_remove, energy_buffer, cumulative, seam);
    double stop = omp_get_wtime();
    printf("Time(full): %f s\n", stop - start);

    free(energy_buffer);
    free(cumulative);
    free(seam);

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char* file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }
    file_type++;  // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, (int)width, (int)height, COLOR_CHANNELS, im, (orig_width * COLOR_CHANNELS));
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);

    return 0;
}
