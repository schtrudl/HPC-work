SIZE ?= 64

ifdef GENERATE_GIF
GIF_FLAG = -DGENERATE_GIF
endif

DEBUG = -fsanitize=address -g

.PHONY: all lenia lenia-ref lenia-mp lenia-mpc lenia-mpt clean

all: lenia lenia-ref lenia-mp lenia-mpc lenia-mpt

lenia-ref:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native --openmp ./src/lenia-ref.c -o lenia-ref.$(SIZE).bin -lm

lenia:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia.c -o lenia.$(SIZE).bin -lm

lenia-mp:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia-mp.c -o lenia-mp.$(SIZE).bin -lm

lenia-mpc:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia-mpc.c -o lenia-mpc.$(SIZE).bin -lm

lenia-mpt:
	mpicc -DSIZE=$(SIZE) $(GIF_FLAG) -O3 -march=native -ffast-math --openmp ./src/lenia-mpt.c -o lenia-mpt.$(SIZE).bin -lm

clean:
	rm -f *.bin