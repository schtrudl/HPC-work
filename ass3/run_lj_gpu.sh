#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_final.log
#SBATCH --time=01:00:00

#LOAD MODULES
module load CUDA

#RUN
./run.py gpu -n=10 --srun
#module load FFmpeg
#./verify.py --srun gpu