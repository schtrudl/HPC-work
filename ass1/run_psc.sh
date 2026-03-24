#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=psc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=timings.final.log
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --mem=128G

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the numactl module to enable numa library linking
module load numactl

rm sample

# Compile
gcc -O3 -march=native -lm -lnuma --openmp sample.c -o sample

for i in 2 4 8 16 32; do
    OMP_NUM_THREADS=$i srun python3 run.py -n 20 >> timings.final$i.log
done
