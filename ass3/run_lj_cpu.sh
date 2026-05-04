#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lj_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --output=timings_cpu_soa.log
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

rm -f lj_cpu lj_cpu2
module load CUDA
./run.py --srun cpu2 -n=10

#module load FFmpeg
#./verify.py --srun cpu2 --full

