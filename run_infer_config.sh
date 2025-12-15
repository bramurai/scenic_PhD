#!/bin/bash
#SBATCH --job-name=infer_config
#SBATCH --output=logs/infer_config_%j.out
#SBATCH --error=logs/infer_config_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source activate scenic_phd

python3 infer_config_from_checkpoint.py
