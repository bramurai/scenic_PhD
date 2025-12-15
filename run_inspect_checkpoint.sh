#!/bin/bash
#SBATCH --job-name=inspect_ckpt
#SBATCH --output=logs/inspect_checkpoint_%j.out
#SBATCH --error=logs/inspect_checkpoint_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "=================================="
echo "Checkpoint Inspection Job"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=================================="

source activate scenic_phd

# Run inspection
cd /project/3026018.01/Models/MBT
python3 inspect_checkpoint.py

echo ""
echo "=================================="
echo "Running parameter verification..."
echo "=================================="
python3 debug_model_params.py

echo "=================================="
echo "Inspection complete!"
echo "=================================="
