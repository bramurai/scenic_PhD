#!/bin/bash
#SBATCH --job-name=vggsound_preprocess
#SBATCH --output=logs/vggsound_%A_%a.out
#SBATCH --error=logs/vggsound_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load singularity/apptainer if available
module load singularity  # or: module load apptainer

# Use a pre-built container with all dependencies
CONTAINER="/path/to/tensorflow_ffmpeg.sif"

# Or pull one if not available:
# singularity pull docker://tensorflow/tensorflow:2.13.0

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
OUTPUT_DIR="/scratch/$USER/vggsound/tfrecords/train"
TEMP_DIR="/scratch/$USER/vggsound/temp_${SLURM_ARRAY_TASK_ID}"
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR

echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"

# Run inside container
singularity exec --bind /scratch:/scratch $CONTAINER bash -c "
    pip install --user yt-dlp librosa absl-py
    python download_and_preprocess_vggsound.py \
      --csv_path=PreProcessing/vggsound_train.csv \
      --output_path=$OUTPUT_DIR \
      --num_shards=100 \
      --temp_dir=$TEMP_DIR \
      --check_duration=True
"

rm -rf $TEMP_DIR
echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
