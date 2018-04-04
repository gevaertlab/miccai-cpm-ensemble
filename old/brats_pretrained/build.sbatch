#!/bin/bash

#SBATCH --job-name=build
#SBATCH --output=build.out
#SBATCH --error=build.err

#SBATCH --time=1:00:00
#SBATCH --mem=2000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow/0.9.0
cd tumor-seg/utils/build_utils

python build.py
