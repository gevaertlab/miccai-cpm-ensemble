#!/bin/bash

#SBATCH --job-name=rembrandt_test
#SBATCH --output=rembrandt_test.out
#SBATCH --error=rembrandt_test.err

#SBATCH --time=1:00:00
#SBATCH --mem=32000

#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/2.7.5
module load tensorflow/0.9.0
cd ~/tumor_seg

python segment.py --cfg-path=config/rembrandt_test.cfg
