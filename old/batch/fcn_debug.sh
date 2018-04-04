#!/bin/bash

#SBATCH --job-name=fcn_debug
#SBATCH --output=fcn_debug.out
#SBATCH --error=fcn_debug.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python fcn_train.py --cfg-path=debug.cfg
