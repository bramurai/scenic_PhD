#!/bin/bash

################################################################################
# SLURM Job Script for MBT Class-Averaged Activation Extraction
# 
# This script runs extract_mbt_activations_class_averaged.py on an HPC cluster
# 
# Usage:
#   sbatch run_mbt_extraction.sh
# 
# To customize, edit the SLURM parameters below (lines starting with #SBATCH)
################################################################################

#SBATCH --job-name=mbt_extraction
#SBATCH --partition=gpu                    # Partition (adjust to your cluster)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=8                  # CPUs per task
#SBATCH --gres=gpu:1                       # GPUs (1 GPU, adjust as needed)
#SBATCH --mem=64G                          # Total memory (64GB recommended)
#SBATCH --time=24:00:00                    # Wall time (24 hours)
#SBATCH --output=logs/mbt_extraction_%j.log     # Output log file (%j = job ID)
#SBATCH --error=logs/mbt_extraction_%j.err      # Error log file

set -e  # Exit on error

################################################################################
# Configuration Variables (Edit these for your setup)
################################################################################

# Path to this script's directory
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$SCRIPT_DIR"

# Optional: Clean old extraction logs before running (keeps only the 5 most recent)
echo "Cleaning old extraction logs (keeping 5 most recent)..."

cd "$SCRIPT_DIR/logs" && \
    ls -t mbt_extraction_*.log mbt_extraction_*.err 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null
cd "$SCRIPT_DIR"
echo ""

# Python environment (choose one):
# Option 1: Conda environment
CONDA_ENV="scenic_phd"
# Option 2: Virtual environment
# VENV_PATH="/path/to/venv"

# MBT Script
MBT_SCRIPT="${SCRIPT_DIR}/extract_mbt_activations_class_averaged.py"

# Config file
CONFIG="${SCRIPT_DIR}/scenic/projects/mbt/configs/audioset/Inference_config.py"

# Checkpoint directory
CHECKPOINT_DIR="${SCRIPT_DIR}/CheckPoints/MBT_AV"

# Test data directory
TEST_DATA_DIR="${SCRIPT_DIR}/Datasets/audioset_eval_configCorrect"

# Labels CSV
LABELS_CSV="${SCRIPT_DIR}/Video_csvs/audioset_labels.csv"

# Output directory
OUTPUT_DIR="${SCRIPT_DIR}/audioset_extraction_$(date +%Y%m%d_%H%M%S)"

# Processing parameters
NUM_SAMPLES=16          # None = process all, or specify number (e.g., 500)
BATCH_SIZE=16              # Batch size (1 = slower but safer for memory)
CHECKPOINT_EVERY=1       # Save checkpoint every N batches (0 = disable)
CLEAR_CACHE_EVERY=1       # Clear JAX cache every N batches

# Flags for what to save/compute
SAVE_ACTIVATIONS="--save_activations"           # --save_activations or --nosave_activations
SAVE_LOGITS="--save_logits"                     # Include or remove
COMPUTE_MAP="--compute_map"                     # Include or remove
SAVE_ATTENTION=""                               # --save_attention (increases storage) or empty
AVERAGE_ATTENTION_HEADS="--average_attention_heads"  # Include or remove

################################################################################
# Setup and Initialization
################################################################################

echo "================================================================================"
echo "MBT Class-Averaged Activation Extraction"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo ""

# Create logs directory if it doesn't exist


# Print environment info
echo "Environment Information:"
echo "  Node: $(hostname)"
echo "  CPUs available: $(nproc)"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPUs available:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    /'
fi
echo ""

# Print configuration
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Test data: $TEST_DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Num samples: $NUM_SAMPLES"
echo ""

################################################################################
# Load Environment and Activate Python
################################################################################

echo "Loading modules and activating Python environment..."

# Load modules (adjust based on your cluster)
# Load CUDA 11.4 or 11.8 (NOT cuda/8.0 or older!)
if module load cuda/11.8 2>/dev/null; then
    echo "  Loaded cuda/11.8"
elif module load cuda/11.4 2>/dev/null; then
    echo "  Loaded cuda/11.4"
else
    echo "  Warning: Could not load CUDA 11.x module"
fi

# Optional: Load cuDNN if available and needed
# module load cudnn/8.6

# Set environment variables to work around cuDNN issues
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
echo "  Set GPU/cuDNN workaround environment variables"

# Activate conda environment
if [ -n "$CONDA_ENV" ]; then
    echo "  Using conda environment: $CONDA_ENV"
    source ~/.bashrc
    conda activate "$CONDA_ENV"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate conda environment '$CONDA_ENV'"
        exit 1
    fi
# Otherwise activate venv
elif [ -n "$VENV_PATH" ]; then
    echo "  Using virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Verify Python
PYTHON=$(which python3)
echo "  Python: $PYTHON"
$PYTHON --version
echo ""

################################################################################
# Create Output Directory
################################################################################

mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

################################################################################
# Pre-processing Checks
################################################################################

echo "Checking required files..."

checks_passed=true

if [ ! -f "$CONFIG" ]; then
    echo "  ERROR: Config file not found: $CONFIG"
    checks_passed=false
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "  ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    checks_passed=false
fi

if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "  ERROR: Test data directory not found: $TEST_DATA_DIR"
    checks_passed=false
fi

if [ ! -f "$LABELS_CSV" ]; then
    echo "  ERROR: Labels CSV not found: $LABELS_CSV"
    checks_passed=false
fi

if [ ! -f "$MBT_SCRIPT" ]; then
    echo "  ERROR: MBT script not found: $MBT_SCRIPT"
    checks_passed=false
fi

if [ "$checks_passed" = false ]; then
    echo "Pre-processing checks FAILED. Exiting."
    exit 1
fi

echo "  All required files found ✓"
echo ""

################################################################################
# Set Environment Variables for JAX (Optional but Recommended)
################################################################################

# Limit JAX memory growth to avoid OOM
export JAX_PLATFORMS="gpu"                    # Use GPU, or "cpu" for CPU-only
export JAX_BACKEND_TARGET="gpu"
# export JAX_DEVICES="cuda:0"                 # Specific GPU device (optional)

# Optional: Set memory pre-allocation percentage (0.0-1.0)
# export JAX_CUDA_VISIBLE_DEVICES="0"         # Use GPU 0

# Disable Jit warnings
export TF_CPP_MIN_LOG_LEVEL=2                 # Suppress TensorFlow info messages

echo "JAX/TensorFlow environment:"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS:-default}"
echo "  TF_CPP_MIN_LOG_LEVEL: ${TF_CPP_MIN_LOG_LEVEL:-default}"
echo ""

################################################################################
# Run the MBT Extraction Script
################################################################################

echo "================================================================================"
echo "Running MBT extraction..."
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Build the command
CMD="$PYTHON $MBT_SCRIPT \
    --config=$CONFIG \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --test_data_dir=$TEST_DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --audioset_labels_csv=$LABELS_CSV \
    --batch_size=$BATCH_SIZE \
    --checkpoint_every=$CHECKPOINT_EVERY \
    --clear_cache_every=$CLEAR_CACHE_EVERY"

# Add conditional flags
if [ "$NUM_SAMPLES" != "None" ]; then
    CMD="$CMD --num_samples=$NUM_SAMPLES"
fi

if [ -n "$SAVE_ACTIVATIONS" ]; then
    CMD="$CMD $SAVE_ACTIVATIONS"
fi

if [ -n "$SAVE_LOGITS" ]; then
    CMD="$CMD $SAVE_LOGITS"
fi

if [ -n "$COMPUTE_MAP" ]; then
    CMD="$CMD $COMPUTE_MAP"
fi

if [ -n "$SAVE_ATTENTION" ]; then
    CMD="$CMD $SAVE_ATTENTION"
fi

if [ -n "$AVERAGE_ATTENTION_HEADS" ]; then
    CMD="$CMD $AVERAGE_ATTENTION_HEADS"
fi

# Print the command that will be executed
echo "Command:"
echo "$CMD"
echo ""

# Run the command
eval "$CMD"
EXTRACTION_EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "Extraction Complete"
echo "================================================================================"
echo "Exit code: $EXTRACTION_EXIT_CODE"
printf "Elapsed time: %d hours, %d minutes, %d seconds\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))
echo "Output directory: $OUTPUT_DIR"
echo ""

################################################################################
# Post-processing and Cleanup
################################################################################

if [ $EXTRACTION_EXIT_CODE -eq 0 ]; then
    echo "✓ Extraction successful!"
    
    # List output files
    echo ""
    echo "Output files created:"
    ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
    
    # Show results summary
    if [ -f "$OUTPUT_DIR/class_statistics.csv" ]; then
        echo ""
        echo "Class statistics saved to: class_statistics.csv"
    fi
    
    if [ -f "$OUTPUT_DIR/averaged_activations/metadata.pkl" ]; then
        echo "Averaged activations saved to: averaged_activations/"
    fi
    
else
    echo "✗ Extraction FAILED with exit code $EXTRACTION_EXIT_CODE"
    echo ""
    echo "Check the error log for details:"
    echo "  logs/mbt_extraction_${SLURM_JOB_ID}.err"
    exit $EXTRACTION_EXIT_CODE
fi

echo ""
echo "Finished at: $(date)"
echo "================================================================================"

exit 0
