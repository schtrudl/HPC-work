#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lj-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus=2
# lahko sta dva
#SBATCH --nodes=1
#SBATCH --output=timings_8_joined.log
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00

#LOAD MODULES
#module load CUDA
module purge
module load NCCL

#export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0  # ensure peer-to-peer is enabled

#make
#srun ./lj.out

./run.py --srun -n 5
#./verify.py --srun --full

#CELL_SKIN_VALUES=${CELL_SKIN_VALUES:-"0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40"}
#
#for skin in ${CELL_SKIN_VALUES}; do
#	CELL_SKIN="${skin}" ./run.py -n 5 --steps 20000 --srun > "lj-skin-${skin}.log"
#done
