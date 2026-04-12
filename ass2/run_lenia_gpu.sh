#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_8_fastmath.log

#LOAD MODULES
module load CUDA

#RUN
./run.py gpu -n=10 --srun
#module load FFmpeg
#./verify.py --srun gpu