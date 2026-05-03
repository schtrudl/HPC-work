#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_6_2D_y_1.log
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --hint=nomultithread

# Set OpenMP environment variables for thread placement and binding
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#LOAD MODULES
module load CUDA

#RUN
./run.py gpu -n=10 --srun
#module load FFmpeg
#./verify.py --srun gpu --full