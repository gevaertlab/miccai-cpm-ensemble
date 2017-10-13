#!/bin/bash

#SBATCH --job-name=bn_train
#SBATCH --output=bn_train.out
#SBATCH --error=bn_train.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python fcn_train.py --cfg-path=bn_train.cfg
