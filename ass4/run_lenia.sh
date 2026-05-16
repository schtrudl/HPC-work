#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=x.log
#SBATCH --hint=nomultithread

#Load MPI module
module load OpenMPI

#Build
make

#Run
#mpirun -np $SLURM_NTASKS ./lenia
module load FFMPEG
./verify.py lenia

