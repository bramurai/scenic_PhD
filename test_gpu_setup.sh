#!/bin/bash

################################################################################
# SLURM Test Job for GPU and MBT Setup Verification
# 
# Quick test (5 min) to verify GPU access and MBT code before running full job
# 
# Usage:
#   sbatch test_gpu_setup.sh
################################################################################

#SBATCH --job-name=mbt_test
#SBATCH --partition=gpu                  # Adjust to your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=8G
#SBATCH --time=00:10:00                  # 10 minutes (should finish in 2-5)
#SBATCH --output=/project/3026018.01/Models/MBT/logs/test_%j.log
#SBATCH --error=/project/3026018.01/Models/MBT/logs/test_%j.err

set -e

# Get the submission directory (where sbatch was called from)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$SCRIPT_DIR"

# Optional: Clean old logs before running (keeps only the 5 most recent)
echo "Cleaning old test logs..."
cd "$SCRIPT_DIR/logs" 2>/dev/null && \
    ls -t test_*.log test_*.err 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "MBT GPU Setup Test"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Working directory: $SCRIPT_DIR"
echo ""



# Load modules if needed (edit based on your cluster)
# Note: Don't use 'module purge' - it may break required modules like cluster/1.0
module load anaconda3

# Try to load CUDA - check which versions are available
# First, list available CUDA modules to help troubleshooting
echo "Checking for available CUDA modules..."
module avail cuda 2>&1 | head -10 || echo "  (module avail output not available)"
echo ""

# Load CUDA 11.4 (default) or 11.8 - NOT cuda/8.0!
if module load cuda/11.8 2>/dev/null; then
    echo "✓ Loaded cuda/11.8"
elif module load cuda/11.4 2>/dev/null; then
    echo "✓ Loaded cuda/11.4 (default)"
else
    echo "⚠ Warning: Could not load CUDA 11.x module"
fi

echo ""

# Set environment variables to work around cuDNN issues
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
echo "Set GPU/cuDNN workaround environment variables"
echo ""

# Also try loading cuDNN if available
# module load cudnn/8.6

echo "Activating scenic_preprocessing environment..."
source activate scenic_phd

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
# Verify Python
python --version
echo "Python check"
python -c "import jax; print(jax.__version__)"
# Run the test script
python3 test_gpu_setup.py

EXIT_CODE=$?

echo ""
echo "Test completed with exit code: $EXIT_CODE"
echo "Finished: $(date)"

exit $EXIT_CODE
