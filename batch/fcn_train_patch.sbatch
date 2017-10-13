#!/bin/bash

#SBATCH --job-name=fcn_patch48_train
#SBATCH --output=fcn_patch48_train.out
#SBATCH --error=fcn_patch48_train.err

#SBATCH --time=12:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python fcn_train.py --cfg-path=fcn_train_patch.cfg
