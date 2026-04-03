#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=timings.cpu.log
#SBATCH --hint=nomultithread

#LOAD MODULES
module load CUDA

#BUILD
make

#RUN
srun ./lenia.out

