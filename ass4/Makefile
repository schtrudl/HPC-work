SIZE ?= 64

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lenia clean

all: lenia

# -ffast-math
lenia:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia.c -o lenia -lm

clean:
	rm -f lenia