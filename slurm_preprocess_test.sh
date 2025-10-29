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


# Directories
OUTPUT_DIR="/scratch/$USER/vggsound_test/tfrecords/train"
TEMP_DIR="/scratch/$USER/vggsound_test/temp_${SLURM_ARRAY_TASK_ID}"
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR
mkdir -p logs

echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"

# Test with just 10 shards processing first 10 videos
python download_and_preprocess_vggsound.py \
  --csv_path=PreProcessing/vggsound_small.csv \
  --output_path=$OUTPUT_DIR \
  --num_shards=1 \
  --temp_dir=$TEMP_DIR \
  --check_duration=True \
  --save_progress_every=5

rm -rf $TEMP_DIR
echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
