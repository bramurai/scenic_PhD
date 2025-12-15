#!/bin/bash
#SBATCH --job-name=verify_tfrecord
#SBATCH --output=logs/verify_tfrecord_%j.out
#SBATCH --error=logs/verify_tfrecord_%j.err
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Verify TFRecord contents and check if they match MBT paper specifications

echo "========================================="
echo "Verifying TFRecords"
echo "========================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source activate scenic_preprocessing

# Set working directory
cd /project/3026018.01/Models/MBT

# Run verification script
python3 verify_tfrecord.py --num_samples=12 --output_dir=logs_tfrecord_verify

echo ""
echo "========================================="
echo "Verification complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Check logs/tfrecord_verify_sample*.png for visualizations"
