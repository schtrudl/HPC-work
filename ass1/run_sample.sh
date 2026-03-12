#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=sample_out.log
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding    
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

# Compile
gcc -O3 -lm -lnuma --openmp sample.c -o sample

# Run
srun  sample valve.png valve-out.png
