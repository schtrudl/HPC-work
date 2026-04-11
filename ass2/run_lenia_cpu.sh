#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=timings_cpu_8.log
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for i in 2 4 8 16 32 64; do
    echo "Run $i"
    export OMP_NUM_THREADS=$i
    ./run.py cpu -n=7 --srun > timings_cpu_c${i}.log
done

#./run.py cpu -n=10 --srun

