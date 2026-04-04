#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=timings_cpu_1_baseline.log
#SBATCH --hint=nomultithread

#BUILD
make lenia_cpu

#RUN
srun ./lenia_cpu

