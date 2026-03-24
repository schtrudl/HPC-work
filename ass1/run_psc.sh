#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=psc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=timings.final1.log
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

rm sample

# Compile
gcc -O3 -march=native -lm -lnuma --openmp sample.c -o sample

srun python3 run.py -n 20
