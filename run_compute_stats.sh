#!/bin/bash
#SBATCH --job-name=compute_stats
#SBATCH --output=logs/compute_stats_%j.out
#SBATCH --error=logs/compute_stats_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source activate scenic_phd

echo "Computing spectrogram statistics from audioset_eval dataset..."
python3 compute_spec_stats.py

echo "Done! Check output for computed mean/stddev values."
