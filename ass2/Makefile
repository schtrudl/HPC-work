# Compiler
CC = nvcc

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp

all: lenia_cpu lenia_gpu

lenia_cpu:
	gcc -O3 -march=native --openmp ./src/lenia_cpu.c -o lenia_cpu -lm

lenia_gpu:
	$(CC) $(CFLAGS) ./src/lenia_gpu.cu -o lenia_gpu $(LDLIBS)

clean:
	rm -f lenia_cpu lenia_gpu