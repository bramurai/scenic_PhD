#!/bin/bash
#SBATCH --job-name=vggsound_test
#SBATCH --output=logs/vggsound_test_%A_%a.out
#SBATCH --error=logs/vggsound_test_%A_%a.err
#SBATCH --array=0-9                     # Just 10 jobs for testing
#SBATCH --time=24:00:00                 # 1 day
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Print debugging info
echo "=== Job Starting ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "User: $USER"
echo "Home: $HOME"
echo ""

# Load required modules (adjust for your cluster)
echo "Loading modules..."
module load anaconda3 2>/dev/null || module load anaconda 2>/dev/null || echo "No anaconda module found, using system conda"

# Initialize conda
echo "Initializing conda..."
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc

# Activate environment
echo "Activating scenic_preprocessing environment..."
conda activate scenic_preprocessing

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE

# Directories (using project directory instead of /scratch)
PROJECT_DIR="$HOME/scenic_PhD"  # Adjust if your project is elsewhere
OUTPUT_DIR="$PROJECT_DIR/PreProcessing/tfrecords_test"
TEMP_DIR="$PROJECT_DIR/PreProcessing/temp_${SLURM_ARRAY_TASK_ID}"

echo "Creating directories..."
echo "Output: $OUTPUT_DIR"
echo "Temp: $TEMP_DIR"
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR
mkdir -p $PROJECT_DIR/logs

# Verify directories were created
if [ -d "$OUTPUT_DIR" ]; then
    echo "✓ Output directory created"
else
    echo "✗ Failed to create output directory"
fi

if [ -d "$TEMP_DIR" ]; then
    echo "✓ Temp directory created"
else
    echo "✗ Failed to create temp directory"
fi
echo ""

# Change to project directory
cd $PROJECT_DIR
echo "Changed to: $(pwd)"
echo "Files here: $(ls -la | head -10)"
echo ""

echo "Job ${SLURM_ARRAY_TASK_ID} starting processing at $(date)"
echo ""

# Test with just 10 shards processing first 10 videos
python download_and_preprocess_vggsound.py \
  --csv_path=PreProcessing/vggsound_small.csv \
  --output_path=$OUTPUT_DIR \
  --num_shards=1 \
  --temp_dir=$TEMP_DIR \
  --check_duration=True \
  --save_progress_every=5

EXIT_CODE=$?
echo ""
echo "Python script exited with code: $EXIT_CODE"

# Cleanup
echo "Cleaning up temp directory..."
rm -rf $TEMP_DIR

echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "=== Job Complete ==="
