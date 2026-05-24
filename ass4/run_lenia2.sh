#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --job-name=lenia
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --output=timings_mpc_1_32_2.log
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:20:00

#Load MPI module
#module load FFmpeg
module load OpenMPI
#mpirun -np $SLURM_NTASKS ./lenia
./run.py -n 3 --binary lenia-mpc
#./verify.py --binary lenia-mpc

#mpicc -O3 -Wall sm.c -o sm -lm
#mpirun --mca pml ob1 -np $SLURM_NTASKS ./sm