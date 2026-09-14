#!/bin/bash

#SBATCH --job-name=nfw_lensing
#SBATCH --nodes=21
#SBATCH --ntasks=1000
#SBATCH --ntasks-per-node=48
#SBATCH --time=00:50:00
#SBATCH --partition=plgrid
#SBATCH --output=nfw_%j.log
#SBATCH --error=nfw_%j.err

cd /net/pr2/projects/plgrid/plgggwlens/codes/mpi
## Initialize conda from your miniconda installation
source /net/pr2/projects/plgrid/plgggwlens/miniconda3/etc/profile.d/conda.sh

# Activate your gwlens environment
conda activate /net/pr2/projects/plgrid/plgggwlens/envs/gwlens

# Verify it worked
echo "Python: $(which python)"
echo "Python version: $(python --version)"

mkdir -p logs

module load openmpi
echo "Job started: $(date)"
mpirun -np $SLURM_NTASKS python /net/pr2/projects/plgrid/plgggwlens/codes/mpi/mpi.py
echo "Job ended: $(date)"
