# Compiler flags
LDLIBS = -lm
CFLAGS = -Wno-deprecated-gpu-targets -gencode=arch=compute_70,code=sm_70 -O3 -Xcompiler --openmp,-march=native

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lj_cpu lj_gpu clean

all: lj_cpu lj_gpu

lj_cpu:
	c++ $(GIF_FLAG) -O3 -march=native --openmp ./src/lj_cpu.cpp -o lj_cpu $(LDLIBS)

lj_gpu:
	nvcc $(GIF_FLAG) $(CFLAGS) ./src/lj_gpu.cu -o lj_gpu $(LDLIBS)

clean:
	rm -f lj_cpu lj_gpu