#!/bin/bash

#SBATCH --job-name=cnn_retrain
#SBATCH --output=cnn_retrain.out
#SBATCH --error=cnn_retrain.err

#SBATCH --time=10:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python cnn_train.py --cfg-path=cnn_retrain.cfg
