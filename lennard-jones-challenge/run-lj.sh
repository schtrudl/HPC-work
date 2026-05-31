#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_8_joined.log
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

#LOAD MODULES
module load CUDA

make
srun ./lj.out

#./run.py --srun -n 1
#./verify.py --srun --full
