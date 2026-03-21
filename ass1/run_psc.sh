#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=psc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=timings.16.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

rm sample

# Compile
gcc -O3 -lm -lnuma --openmp sample.c -o sample

# Run
srun python3 run.py -n 20
