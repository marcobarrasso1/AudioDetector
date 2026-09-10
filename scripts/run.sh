#!/bin/bash
#SBATCH --job-name=mcdc_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=test.txt
#SBATCH --error=test.txt

source ../newenv/bin/activate 

module load cuda 

python test.py --ckpt_path=mcdc_epoch36.pt