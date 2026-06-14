#! /bin/bash -l

## Job name
#SBATCH -J amoore

## Allocate N nodes
#SBATCH -N 1

## ntasks per node (= number of processes = number of cores)
#SBATCH --ntasks-per-node=1

## memory per core
#SBATCH --mem-per-cpu=500MB

## maximum time (HH:MM:SS)

#SBATCH --time=2-00:00:00

## partition (queue) to use
#SBATCH -p short
#SBATCH --output="stdout.txt"
#SBATCH --error="stderr.txt"
## Use account (requred only if different from default camk)
##send an email when done
#SBATCH --mail-type=ALL

module load mpi/mvapich2-x86_64
## jobs starts in $HOME; go to the submission directory
cd $SLURM_SUBMIT_DIR
# run code redirecting stdout and stderr to a file

source /work/sreekanth/miniconda3/bin/activate

conda activate /work/sreekanth/miniconda3/envs/glow_env


python amoore.py
