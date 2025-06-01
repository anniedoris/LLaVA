#!/bin/bash
#SBATCH -n 64
#SBATCH -p pi_faez
#SBATCH --gres=gpu:4
#SBATCH --mem=200000
#SBATCH -t 12:00:00
#SBATCH -o synthbalout.log
#SBATCH -e synthbalerror.log
#SBATCH -w node2435  # force specific node

# Load any necessary modules
# module load anaconda or whatever you need
module load miniforge/23.11.0-0
module load cuda/12.4.0

# Activate your environment
source ~/.bashrc
conda activate /home/adoris/.conda/envs/llava_train

# Run your script
./scripts/v1_5/finetune_h100_synthbal.sh