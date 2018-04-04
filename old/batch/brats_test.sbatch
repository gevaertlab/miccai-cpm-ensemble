#!/bin/bash

#SBATCH --job-name=brats_test
#SBATCH --output=brats_test.out
#SBATCH --error=brats_test.err

#SBATCH --time=1:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tensorflow/0.9.0
cd tumor-seg

python segment.py --cfg-path=config/brats_test.cfg
