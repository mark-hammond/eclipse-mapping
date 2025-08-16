#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 8      # tasks requested
#SBATCH -c 8      # cpus per task
#SBATCH --mem=10000  # memory in Mb
#SBATCH --output=O-%x.%j.out
#SBATCH --error=E-%x.%j.err

mamba activate starry
python paper_eclipse_suite.py gcm_medprec
