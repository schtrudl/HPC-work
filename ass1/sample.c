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
#define COLOR_CHANNELS 0
#define MAX_FILENAME 255
#define SEAMS 128
// this is to prevent formating of the OMP pragmas
#define OMP(x) _Pragma(omp #x)

void copy_image(unsigned char* image_out, const unsigned char* image_in, const size_t size) {
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
        for (size_t i = 0; i < size; ++i) {
            image_out[i] = image_in[i];
        }
    }
}

void compute_energy(const unsigned char* image_in, const size_t width, const size_t height, const size_t channels, unsigned char* energy) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            for (size_t channel = 0; channel < channels; channel++) {
                size_t idx = (row * width + col);
                size_t image_idx = idx * channels + channel;
                energy[idx] = image_in[image_idx];
            }
        }
    }
}

void compute_commulative(const unsigned char* energy, const size_t width, const size_t height, const size_t channels, unsigned char* commulative) {
    // TODO(perf): we can avoid * by computing idx as part of the loop
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            size_t idx = (row * width + col);
            commulative[idx] = energy[idx];
        }
    }
}

void find_seam(const unsigned char* commulative, const size_t width, const size_t height, size_t n_seams_to_remove, size_t* seam) {
    // sprehodi po commulativi in poišče stolpce z najmanjšimi energijami
    // shrani jih v seam (wxs)
}

void remove_seam(const unsigned char* image_in, const size_t width, const size_t height, const size_t channels, const size_t* seam, unsigned char* image_out) {
    // odstrani stolpce podane v seams iz slike in shrani v image_out
}

void main_algo(const unsigned char* image_in, const size_t width, const size_t height, const size_t channels, unsigned char* image_out) {
    const size_t datasize = width * height * sizeof(unsigned char);
    unsigned char* energy_buffer = (unsigned char*)malloc(datasize);
    unsigned char* commulative = (unsigned char*)malloc(datasize);
    compute_energy(image_in, width, height, channels, energy_buffer);
    compute_commulative(energy_buffer, width, height, channels, commulative);
    size_t n_seams = 1;
    size_t* seam = (size_t*)malloc(n_seams * sizeof(size_t));
    find_seam(commulative, width, height, n_seams, seam);
    remove_seam(image_in, width, height, channels, seam, image_out);
    free(energy_buffer);
    free(commulative);
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
    int width, height, cpp;
    unsigned char* image_in = stbi_load(image_in_name, &width, &height, &cpp, COLOR_CHANNELS);

    if (image_in == NULL) {
        printf("Error reading loading image %s!\n", image_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d with %d channels.\n", image_in_name, width, height, cpp);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char* image_out = (unsigned char*)malloc(datasize);
    if (image_out == NULL) {
        printf("Error: Failed to allocate memory for output image!\n");
        stbi_image_free(image_in);
        exit(EXIT_FAILURE);
    }

    // Copy the input image into output and mesure execution time
    double start = omp_get_wtime();
    main_algo(image_in, (size_t)width, (size_t)height, (size_t)cpp, image_out);
    double stop = omp_get_wtime();
    printf("Time to copy: %f s\n", stop - start);

    // Write the output image to file
    char image_out_name_temp[MAX_FILENAME];
    strncpy(image_out_name_temp, image_out_name, MAX_FILENAME);

    const char* file_type = strrchr(image_out_name, '.');
    if (file_type == NULL) {
        printf("Error: No file extension found!\n");
        stbi_image_free(image_in);
        stbi_image_free(image_out);
        exit(EXIT_FAILURE);
    }
    file_type++;  // skip the dot

    if (!strcmp(file_type, "png"))
        stbi_write_png(image_out_name, width, height, cpp, image_out, width * cpp);
    else if (!strcmp(file_type, "jpg"))
        stbi_write_jpg(image_out_name, width, height, cpp, image_out, 100);
    else if (!strcmp(file_type, "bmp"))
        stbi_write_bmp(image_out_name, width, height, cpp, image_out);
    else
        printf("Error: Unknown image format %s! Only png, jpg, or bmp supported.\n", file_type);

    // Release the memory
    stbi_image_free(image_in);
    stbi_image_free(image_out);

    return 0;
}
