SIZE ?= 64

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lenia clean

all: lenia

lenia:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia.c -o lenia -lm

clean:
	rm -f lenia