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
#define REPORT_SUB_TIMES 1
#ifdef REPORT_SUB_TIMES
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
#define SEAMS 128
// koliko seamov bomo obdelali na enkrat
//
// po definiciji problema naj bi bila kle 1
#define SEAMS_PER_ROUND 2
// this is to prevent formating of the OMP pragmas
#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

// typed malloc
#define box(n, type) (type*)malloc((n) * sizeof(type))
// type aliases for better readability
#define usize size_t
#define u8 uint8_t
#define u32 uint32_t
#define f64 double
#define f32 float

// this struct makes it easier to work with the image data
// as c indexing automatically takes into account sizeof(Pixel)
typedef struct {
    u8 r;
    u8 g;
    u8 b;
} Pixel;

inline usize min(usize a, usize b) {
    return (a < b) ? a : b;
}

inline usize min_3(usize a, usize b, usize c) {
    return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

inline usize min_col(usize a, usize b, u32 a_val, u32 b_val) {
    return (a_val < b_val) ? a : b;
}

inline usize min_col_3(usize a, usize b, usize c, u32 a_val, u32 b_val, u32 c_val) {
    return (a_val < b_val) ? ((a_val < c_val) ? a : c) : ((b_val < c_val) ? b : c);
}

void copy_image(Pixel* image_out, const Pixel* image_in, const usize size) {
    OMP(parallel)  //
    {
        // Print thread, CPU, and NUMA node information
        OMP(single)  //
        printf("Using %d threads.\n", omp_get_num_threads());

        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);
        OMP(critical)  //
        printf("Thread %d -> CPU %d NUMA %d\n", tid, cpu, node);

        // Copy the image data in parallel
        OMP(for)  //
        for (usize i = 0; i < size; ++i) {
            image_out[i] = image_in[i];
        }
    }
}

void compute_energy(const Pixel* image, const usize width, const usize stride, const usize height, u8* energy) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    OMP(parallel for)  //
    for (usize row = 0; row < height; row++) {
        usize row_minus_1 = (row == 0) ? 0 : row - 1;
        usize row_plus_1 = (row == height - 1) ? height - 1 : row + 1;
        for (usize col = 0; col < width; col++) {
            usize col_minus_1 = (col == 0) ? 0 : col - 1;
            usize col_plus_1 = (col == width - 1) ? width - 1 : col + 1;
            //Pixel p = image[row * stride + col];
            /*
If the input image pixel at row i and column j is denoted by s ( i , j ) , then the energy E ( i , j ) is computed using Sobel as:

G x = − s ( i − 1 , j − 1 ) − 2 s ( i , j − 1 ) − s ( i + 1 , j − 1 ) + s ( i − 1 , j + 1 ) + 2 s ( i , j + 1 ) + s ( i + 1 , j + 1 )

G y = + s ( i − 1 , j − 1 ) + 2 s ( i − 1 , j ) + s ( i − 1 , j + 1 ) − s ( i + 1 , j − 1 ) − 2 s ( i + 1 , j ) − s ( i + 1 , j + 1 ) 
            */
            // TODO(perf): optimize this for cache
            // TODO(perf): SIMD
            f64 Gxr = -image[row_minus_1 * stride + col_minus_1].r - 2 * image[row * stride + col_minus_1].r - image[row_plus_1 * stride + col_minus_1].r + image[row_minus_1 * stride + col_plus_1].r + 2 * image[row * stride + col_plus_1].r + image[row_plus_1 * stride + col_plus_1].r;
            f64 Gyr = image[row_minus_1 * stride + col_minus_1].r + 2 * image[row_minus_1 * stride + col].r + image[row_minus_1 * stride + col_plus_1].r - image[row_plus_1 * stride + col_minus_1].r - 2 * image[row_plus_1 * stride + col].r - image[row_plus_1 * stride + col_plus_1].r;

            f64 Gxg = -image[row_minus_1 * stride + col_minus_1].g - 2 * image[row * stride + col_minus_1].g - image[row_plus_1 * stride + col_minus_1].g + image[row_minus_1 * stride + col_plus_1].g + 2 * image[row * stride + col_plus_1].g + image[row_plus_1 * stride + col_plus_1].g;
            f64 Gyg = image[row_minus_1 * stride + col_minus_1].g + 2 * image[row_minus_1 * stride + col].g + image[row_minus_1 * stride + col_plus_1].g - image[row_plus_1 * stride + col_minus_1].g - 2 * image[row_plus_1 * stride + col].g - image[row_plus_1 * stride + col_plus_1].g;

            f64 Gxb = -image[row_minus_1 * stride + col_minus_1].b - 2 * image[row * stride + col_minus_1].b - image[row_plus_1 * stride + col_minus_1].b + image[row_minus_1 * stride + col_plus_1].b + 2 * image[row * stride + col_plus_1].b + image[row_plus_1 * stride + col_plus_1].b;
            f64 Gyb = image[row_minus_1 * stride + col_minus_1].b + 2 * image[row_minus_1 * stride + col].b + image[row_minus_1 * stride + col_plus_1].b - image[row_plus_1 * stride + col_minus_1].b - 2 * image[row_plus_1 * stride + col].b - image[row_plus_1 * stride + col_plus_1].b;

            energy[row * width + col] = (u8)((sqrt(Gxr * Gxr + Gyr * Gyr) + sqrt(Gxg * Gxg + Gyg * Gyg) + sqrt(Gxb * Gxb + Gyb * Gyb)) / 3.0);
        }
    }
}

void compute_cumulative(const u8* energy, const usize width, const usize height, u32* cumulative) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    // TODO(perf): implement parallel with dependency triangles

    OMP(parallel for)  //
    for (usize row = 0; row < height; row++) {
        for (usize col = 0; col < width; col++) {
            usize idx = (row * width + col);
            if (row == 0) {
                cumulative[idx] = energy[idx];
                continue;
            }

            if (col == 0) {
                cumulative[idx] = energy[idx] + min(cumulative[idx - width], cumulative[idx - width + 1]);
            } else if (col == width - 1) {
                cumulative[idx] = energy[idx] + min(cumulative[idx - width - 1], cumulative[idx - width]);
            } else {
                cumulative[idx] = energy[idx] + min_3(cumulative[idx - width - 1], cumulative[idx - width], cumulative[idx - width + 1]);
            }
        }
    }
}

// sprehodi po commulativi in poišče stolpce z najmanjšimi energijami
// shrani jih v seam (height * n_seams_to_remove) in vrne kok seamov je naredil
// lahko jih je manj bo pa vsaj en
size_t find_seam(const u32* cumulative, const usize width, const usize height, usize n_seams_to_remove, usize* seam) {
    // TODO: (perf)implement for n > 1 seams
    u32 minimum = cumulative[0];
    usize min_column = 0;

    OMP(parallel for)  //
    for (usize col = 0; col < width; col++) {
        if (cumulative[col] < minimum) {
            minimum = cumulative[col];
            min_column = col;
        }
    }
    seam[0] = min_column;

    for (usize row = 1; row < height; row++) {
        if (seam[row - 1] == 0) {
            seam[row] = min_col(0, 1, cumulative[row * width], cumulative[row * width + 1]);
        } else if (seam[row - 1] == width - 1) {
            seam[row] = min_col(width - 2, width - 1, cumulative[row * width + width - 2], cumulative[row * width + width - 1]);
        } else {
            seam[row] = min_col_3(seam[row - 1] - 1, seam[row - 1], seam[row - 1] + 1, cumulative[row * width + seam[row - 1] - 1], cumulative[row * width + seam[row - 1]], cumulative[row * width + seam[row - 1] + 1]);
        }
    }

    return 1;
}

void remove_seam(Pixel* image, const usize width, const usize stride, const usize height, const usize n_seams, const usize* seam) {
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
            usize idx = (row * stride + col);
            image[idx - s] = image[idx];
        }
    }
}

void main_algo(Pixel* image, usize* width, const usize height, u8* energy_buffer, u32* cumulative, usize* seam) {
    usize stride = *width;
    usize n_seams_to_remove = SEAMS;
    while (n_seams_to_remove > 0) {
        report_time("compute_energy", compute_energy(image, *width, stride, height, energy_buffer));
        report_time("compute_cumulative", compute_cumulative(energy_buffer, *width, height, cumulative));
        usize s;
        report_time_into_var(s, "find_seam", find_seam(cumulative, *width, height, min(SEAMS_PER_ROUND, n_seams_to_remove), seam));
        report_time("remove_seam", remove_seam(image, *width, stride, height, s, seam));
        n_seams_to_remove -= s;
        *width -= s;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }

    char image_in_name[MAX_FILENAME];
    char image_out_name[MAX_FILENAME];

    snprintf(image_in_name, MAX_FILENAME, "%s", argv[1]);
    snprintf(image_out_name, MAX_FILENAME, "%s", argv[2]);

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

    u8* energy_buffer = box(width * height, u8);
    u32* cumulative = box(width * height, u32);
    usize* seam = box(height * SEAMS_PER_ROUND, usize);

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    main_algo(image_in, &width, height, energy_buffer, cumulative, seam);
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
        stbi_write_png(image_out_name, (int)width, (int)height, COLOR_CHANNELS, image_in, (orig_width * COLOR_CHANNELS));
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);

    return 0;
}
