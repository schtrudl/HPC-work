# Compiler
CC = nvcc

SIZE ?= 256

BLOCK_SIZE ?= 16
BLOCK_SIZE_X ?= $(BLOCK_SIZE)
BLOCK_SIZE_Y ?= $(BLOCK_SIZE)

# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp
#CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 --use_fast_math -Xcompiler --openmp,-march=native,-ffast-math

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lenia_cpu lenia_gpu clean

all: lenia_cpu lenia_gpu

lenia_cpu:
	gcc -ffast-math -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia_cpu.c -o lenia_cpu -lm

lenia_gpu:
	$(CC) -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y) -DSIZE=$(SIZE) $(GIF_FLAG) $(CFLAGS) ./src/lenia_gpu.cu -o lenia_gpu $(LDLIBS)

clean:
	rm -f lenia_cpu lenia_gpu