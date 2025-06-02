#!/bin/bash
#SBATCH -n 10
#SBATCH -N 1
#SBATCH -p newnodes
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zimi@mit.edu
python ray_trace.py
