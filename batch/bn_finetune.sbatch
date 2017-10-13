#!/bin/bash

#SBATCH --job-name=bn_finetune
#SBATCH --output=bn_finetune.out
#SBATCH --error=bn_finetune.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow.1/1.1.0
cd ~/tumor_seg

python fcn_finetune.py --cfg-path=bn_finetune.cfg
