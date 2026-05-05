#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lj_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=timings_cpu_skin_x.log
#SBATCH --time=01:00:00

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#for i in 64 32 16 8 4 2 1; do
#    echo "Run $i"
#    export OMP_NUM_THREADS=$i
#    ./run.py cpu -n=5 --srun > timings_cpu_c${i}.log
#done

# 0.05 is bad
# 0.10 is good
# 0.15 is bad
#for i in 0.06 0.07 0.08 0.09 0.11 0.12 0.13 0.14; do
#    echo "Run with CELL_SKIN=$i"
#    CELL_SKIN=$i ./run.py --srun cpu -n=10 > timings_cpu_skin_${i}.log
#done

#module load CUDA
./run.py --srun cpu -n=10

#module load FFmpeg
#./verify.py --srun cpu --full

