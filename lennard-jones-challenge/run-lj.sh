#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=x.log
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

#LOAD MODULES
module load CUDA

#make
#srun ./lj.out

./verify.py --srun --full
