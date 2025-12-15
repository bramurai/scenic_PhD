#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=logs/grid_search_%j.out
#SBATCH --error=logs/grid_search_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source activate scenic_phd

echo "Starting grid search for inference parameters..."
echo "Testing 3 num_frames × 2 spec_mean × 2 spec_stddev × 2 fusion_layer = 24 configs"
echo "Each config takes ~2-3 minutes (Pass 1 only), total ~1 hour"
echo ""

python3 grid_search_inference_params.py

echo ""
echo "Grid search complete! Check grid_search_results/grid_search_results.json"
