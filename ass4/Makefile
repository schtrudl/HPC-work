SIZE ?= 64

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

DEBUG = -fsanitize=address -g

.PHONY: all lenia lenia-ref lenia-mp clean

all: lenia lenia-ref lenia-mp

lenia-ref:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia-ref.c -o lenia-ref -lm

lenia:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia.c -o lenia -lm

lenia-mp:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia-mp.c -o lenia-mp -lm

clean:
	rm -f lenia lenia-ref lenia-mp