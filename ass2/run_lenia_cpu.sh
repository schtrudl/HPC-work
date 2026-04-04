#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=timings_cpu_1_baseline.log
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#RUN
./run.py cpu -n=3

