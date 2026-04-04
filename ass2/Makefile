# Compiler
CC = nvcc

N ?= 256

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp

.PHONY: all lenia_cpu lenia_gpu clean

all: lenia_cpu lenia_gpu

lenia_cpu:
	gcc -DN=$(N) -O3 -march=native --openmp ./src/lenia_cpu.c -o lenia_cpu -lm

lenia_gpu:
	$(CC) -DN=$(N) $(CFLAGS) ./src/lenia_gpu.cu -o lenia_gpu $(LDLIBS)

clean:
	rm -f lenia_cpu lenia_gpu