#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --output=lenia_out.log
#SBATCH --hint=nomultithread

#Load MPI module
module load OpenMPI

#Build
make

#Run
mpirun -np $SLURM_NTASKS ./lenia

