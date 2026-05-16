#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=timings_1_baseline.log
#SBATCH --hint=nomultithread
#SBATCH --time=10:10:00

#Load MPI module
module load OpenMPI

#mpirun -np $SLURM_NTASKS ./lenia
./run.py -n 3
#module load FFmpeg
#./verify.py

