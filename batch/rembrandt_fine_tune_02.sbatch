#!/bin/bash

#SBATCH --job-name=rembrandt_fine_tune_02
#SBATCH --output=rembrandt_fine_tune_02.out
#SBATCH --error=rembrandt_fine_tune_02.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow/0.9.0
cd ~/tumor_seg

python fine_tune.py --cfg-path=config/rembrandt_fine_tune_02.cfg
