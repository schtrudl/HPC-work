#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=timings_gpu_x.log
#SBATCH --time=01:00:00

#LOAD MODULES
module load CUDA

# x*y <= 1024
# x and y should be divisible by 32 for best performance, but we can also test non-square blocks
# BLOCK_SIZES=(
#   # 1xY
#   "1 2" "1 4" "1 8" "1 16" "1 32" "1 64" "1 128" "1 256"
#   # 2xY
#   "2 2" "2 4" "2 8" "2 16" "2 32" "2 64" "2 128" "2 256"
#   # 4xY
#   "4 2" "4 4" "4 8" "4 16" "4 32" "4 64" "4 128" "4 256"
#   # 8xY
#   "8 2" "8 4" "8 8" "8 16" "8 32" "8 64" "8 128"
#   # 16xY
#   "16 2" "16 4" "16 8" "16 16" "16 32" "16 64"
#   # 32xY
#   "32 2" "32 4" "32 8" "32 16" "32 32"
#   # 64xY
#   "64 2" "64 4" "64 8" "64 16"
#   # 128xY
#   "128 2" "128 4" "128 8"
#   # 256xY
#   "256 2" "256 4"
# )

# for bs in "${BLOCK_SIZES[@]}"; do
#   read BLOCK_SIZE_X BLOCK_SIZE_Y <<< "$bs"
#   export BLOCK_SIZE_X
#   export BLOCK_SIZE_Y

#   echo "Running with ${BLOCK_SIZE_X}x${BLOCK_SIZE_Y}"

#   ./run.py gpu -n=10 --srun > timings_gpu_block${BLOCK_SIZE_X}x${BLOCK_SIZE_Y}.log
# done

#RUN
./run.py gpu -n=10 --srun
#module load FFmpeg
#./verify.py --srun gpu