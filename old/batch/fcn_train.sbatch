#!/bin/bash

#SBATCH --job-name=fcn_retrain
#SBATCH --output=fcn_retrain.out
#SBATCH --error=fcn_retrain.err

#SBATCH --time=10:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python fcn_train.py --cfg-path=fcn_retrain.cfg
