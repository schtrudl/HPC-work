#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=x.log
#SBATCH --hint=nomultithread

#Load MPI module
module load OpenMPI

#mpirun -np $SLURM_NTASKS ./lenia
#./run.py
module load FFmpeg
./verify.py

