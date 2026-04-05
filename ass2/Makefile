# Compiler
CC = nvcc

N ?= 256

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lenia_cpu lenia_gpu clean

all: lenia_cpu lenia_gpu

lenia_cpu:
	gcc -DN=$(N) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia_cpu.c -o lenia_cpu -lm

lenia_gpu:
	$(CC) -DN=$(N) $(GIF_FLAG) $(CFLAGS) ./src/lenia_gpu.cu -o lenia_gpu $(LDLIBS)

clean:
	rm -f lenia_cpu lenia_gpu