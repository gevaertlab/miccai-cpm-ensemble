#!/bin/bash

#SBATCH --job-name=cnn_finetune_new
#SBATCH --output=cnn_finetune_new.out
#SBATCH --error=cnn_finetune_new.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python cnn_finetune.py --cfg-path=cnn_finetune.cfg
