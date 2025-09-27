#!/bin/bash

#SBATCH --job-name=grackle
#SBATCH --time=00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zimi@mit.edu     ## CHANGE THIS!!

export PYTHONUNBUFFERED=1
export HDF5_DISABLE_VERSION_CHECK=1

source ~/.bashrc
conda activate grackle-env
#python -c 'import pygrackle'

python grackle_script.py /ceph/submit/data/user/z/zimi/analysis/FIRE/m12f_res7100 20 ambient
