#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sched.h>
#include <numa.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Use 0 to retain the original number of color channels
#define COLOR_CHANNELS 3  // XXX: can we assume this
#define MAX_FILENAME 255
#define SEAMS 128
// this is to prevent formating of the OMP pragmas
#define OMP(x) DO_PRAGMA(omp x)
#define DO_PRAGMA(x) _Pragma(#x)

// typed malloc
#define box(n, type) (type*)malloc((n) * sizeof(type))
// type aliases for better readability
#define usize size_t
#define u8 u_int8_t

// this struct makes it easier to work with the image data
// as c indexing automatically takes into account sizeof(Pixel)
typedef struct {
    u8 r;
    u8 g;
    u8 b;
} Pixel;

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
    for (usize row = 0; row < height; row++) {
        for (usize col = 0; col < width; col++) {
            // TODO(perf): can we always assume 3 channels? anyway lets hope that compiler will unroll the loop
            size_t energy_sum = 0;  // TODO: this type is probably wrong
            Pixel p = image[row * stride + col];
            energy_sum += p.r;
            energy_sum += p.g;
            energy_sum += p.b;
            energy[row * width + col] = energy_sum / 3;
        }
    }
}

void compute_cumulative(const u8* energy, const usize width, const usize height, u8* cumulative) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    for (usize row = 0; row < height; row++) {
        for (usize col = 0; col < width; col++) {
            usize idx = (row * width + col);
            cumulative[idx] = energy[idx];
        }
    }
}

// sprehodi po commulativi in poišče stolpce z najmanjšimi energijami
// shrani jih v seam (height * n_seams_to_remove) in vrne kok seamov je naredil
// lahko jih je manj bo pa vsaj en
size_t find_seam(const u8* cumulative, const usize width, const usize height, usize n_seams_to_remove, usize* seam) {
    // TODO: actual impl, currently we just remove first and middle

    for (usize i = 0; i < height; i++) {
        // needs to be ordered for remove_seam to work correctly (and efficiently)
        seam[i * n_seams_to_remove + 0] = 0;
        seam[i * n_seams_to_remove + 1] = width / 2;
    }
    return 2;
}

void remove_seam(Pixel* image, const usize width, const usize stride, const usize height, const usize n_seams, const usize* seam) {
    if (n_seams == 0) return;
    // odstrani stolpce podane v seams iz slike in shrani v image_out
    // OMP(parallel for) // TODO(perf): paralization probably not worth for small images
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

usize min(usize a, usize b) {
    return (a < b) ? a : b;
}

// TODO(perf): edit image in-place
void main_algo(Pixel* image, usize* width, const usize height, u8* energy_buffer, u8* cumulative) {
    usize stride = *width;
    // koliko seamov bomo obdelali na enkrat
    usize n_seams_to_remove = SEAMS;
    usize n_seams_per_round = 2;  // po definiciji problema naj bi bila kle 1
    usize* seam = box(height * n_seams_per_round, usize);  // XXX: should we move alloc outside of timing?
    while (n_seams_to_remove > 0) {
        compute_energy(image, *width, stride, height, energy_buffer);
        compute_cumulative(energy_buffer, *width, height, cumulative);
        usize s = find_seam(cumulative, *width, height, min(n_seams_per_round, n_seams_to_remove), seam);
        remove_seam(image, *width, stride, height, s, seam);
        n_seams_to_remove -= s;
        *width -= s;
    }
    free(seam);
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
    u8* cumulative = box(width * height, u8);

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    main_algo(image_in, &width, height, energy_buffer, cumulative);
    double stop = omp_get_wtime();
    printf("Time to copy: %f s\n", stop - start);

    free(energy_buffer);
    free(cumulative);

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

    // XXX: can we assume png
    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, (int)width, (int)height, COLOR_CHANNELS, image_in, (orig_width * COLOR_CHANNELS));
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);

    return 0;
}
