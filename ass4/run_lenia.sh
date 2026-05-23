#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=2
#SBATCH --output=timings_basic_row_2_64.log
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:10:00

#Load MPI module
#module load FFmpeg
module load OpenMPI
#mpirun -np $SLURM_NTASKS ./lenia
./run.py -n 5 --binary lenia-mp
#./verify.py --binary lenia-mp
rm -rf *.token

#mpicc -O3 -Wall sm.c -o sm -lm
#mpirun --mca pml ob1 -np $SLURM_NTASKS ./sm