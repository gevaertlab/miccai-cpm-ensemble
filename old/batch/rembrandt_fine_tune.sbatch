#!/bin/bash

#SBATCH --job-name=rembrandt_fine_tune
#SBATCH --output=rembrandt_fine_tune.out
#SBATCH --error=rembrandt_fine_tune.err

#SBATCH --time=6:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow
cd ~/tumor_seg

python fine_tune.py --cfg-path=config/rembrandt_fine_tune_05.cfg
