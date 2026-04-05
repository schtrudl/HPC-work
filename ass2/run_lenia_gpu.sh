#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_1_baseline.log

#LOAD MODULES
module load CUDA

#RUN
./run.py gpu -n=20 --srun
