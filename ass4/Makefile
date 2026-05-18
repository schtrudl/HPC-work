SIZE ?= 64

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

.PHONY: all lenia lenia-ref clean

all: lenia lenia-ref

lenia-ref:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia-ref.c -o lenia-ref -lm

lenia:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia.c -o lenia -lm

clean:
	rm -f lenia lenia-ref