#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_13_tiling_on_gpu.log
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#LOAD MODULES
module load CUDA

#for block_size in 32 64 128 256 512; do
#    echo "Run with block size $block_size"
#    BLOCK_SIZE=$block_size ./run.py --srun gpu -n=10 > timings_gpu_b${block_size}.log
#done

#RUN
./run.py gpu -n=10 --srun
#module load FFmpeg
#./verify.py --srun gpu --full