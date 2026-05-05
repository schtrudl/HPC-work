# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp,-march=native
AFLAGS =

ifdef GENERATE_GIF
AFLAGS += -DGENERATE_GIF
endif

ifdef CELL_SKIN
AFLAGS += -DCELL_SKIN=$(CELL_SKIN)
endif

ifdef BLOCK_SIZE
AFLAGS += -DBLOCK_SIZE=$(BLOCK_SIZE)
endif

.PHONY: all lj_cpu lj_gpu lj_cpu2 clean

all: lj_cpu lj_gpu lj_cpu2

lj_cpu:
	c++ $(AFLAGS) -O3 -march=native --openmp ./src/lj_cpu.cpp -o lj_cpu $(LDLIBS)

lj_gpu:
	nvcc $(AFLAGS) $(CFLAGS) ./src/lj_gpu.cu -o lj_gpu $(LDLIBS)

lj_cpu2:
	nvcc $(AFLAGS) -DLJ_GPU=0 $(CFLAGS) ./src/lj_gpu.cu -o lj_cpu2 $(LDLIBS)

clean:
	rm -f lj_cpu lj_gpu lj_cpu2