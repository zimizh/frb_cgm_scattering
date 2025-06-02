#!/bin/bash

#SBATCH --job-name=grackle
#SBATCH -p mit_quicktest
#SBATCH --time=00:14:50
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zimi@mit.edu     ## CHANGE THIS!!

module load deprecated-modules
module load python/3.10.8-x86_64
## module load gsl/2.5              
module load hdf5/1.14.3
module load openmpi/5.0.6
module load gcc/12.2.0

export PYTHONUNBUFFERED=1
export HDF5_DISABLE_VERSION_CHECK=1

source /pool001/zimi/grackle-env/bin/activate
#python -c 'import pygrackle'

python engaging_grackle_script.py /pool001/zimi/analysis/FIRE/m12f_res7100 20 
