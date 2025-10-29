#!/bin/bash
# Setup script for VGGSound preprocessing on cluster
# Run this once before submitting jobs

set -e  # Exit on error

echo "=== VGGSound Preprocessing Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Load anaconda module first:"
    echo "  module load anaconda3"
    exit 1
fi

echo "1. Creating conda environment..."
conda create -n scenic_preprocessing python=3.10 -y

echo "2. Activating environment..."
source activate scenic_preprocessing

echo "3. Installing conda packages..."
conda install -c conda-forge ffmpeg pandas numpy absl-py -y

echo "4. Installing pip packages..."
pip install --no-cache-dir yt-dlp librosa tensorflow==2.13.0 keras-preprocessing tensorflow-io-gcs-filesystem

echo "5. Setting environment variable..."
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
# Reactivate to apply the variable
source activate scenic_preprocessing

echo "6. Verifying installation..."
python -c "import yt_dlp; print(f'yt-dlp: {yt_dlp.version.__version__}')"
python -c "import librosa; print(f'librosa: {librosa.__version__}')"
python -c "import tensorflow as tf; print(f'tensorflow: {tf.__version__}')"
ffmpeg -version | head -n 1

echo ""
echo "7. Creating directories..."
mkdir -p logs
mkdir -p /scratch/$USER/vggsound/tfrecords/train

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Submit test job: sbatch slurm_preprocess_test.sh"
echo "2. Check logs: tail -f logs/vggsound_test_*.out"
echo "3. If successful, submit full job: sbatch slurm_preprocess.sh"
