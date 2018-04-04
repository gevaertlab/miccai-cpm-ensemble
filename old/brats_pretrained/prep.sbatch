#!/bin/bash

#SBATCH --job-name=prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err

#SBATCH --time=1:00:00
#SBATCH --mem=2000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow/0.9.0
cd tumor-seg/utils/

python prep_brats_data_full.py
