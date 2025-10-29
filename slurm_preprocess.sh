#!/bin/bash
#SBATCH --job-name=vggsound_preprocess
#SBATCH --output=logs/vggsound_%A_%a.out
#SBATCH --error=logs/vggsound_%A_%a.err
#SBATCH --array=0-99                    # 100 parallel jobs (one per shard)
#SBATCH --time=72:00:00                 # 3 days per job
#SBATCH --mem=16G                       # 16GB RAM per job
#SBATCH --cpus-per-task=4               # 4 CPUs per job
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules (adjust for your cluster)
module load anaconda3
# Note: ffmpeg is installed in the conda environment, no module needed

# Activate conda environment (ffmpeg is included)
source activate scenic_preprocessing

# Set OpenMP workaround
export KMP_DUPLICATE_LIB_OK=TRUE

# Create output and temp directories
OUTPUT_DIR="/scratch/$USER/vggsound/tfrecords/train"
TEMP_DIR="/scratch/$USER/vggsound/temp_${SLURM_ARRAY_TASK_ID}"
mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR
mkdir -p logs

# Calculate which rows this job should process
# Each job processes rows where (row_index % 100) == SLURM_ARRAY_TASK_ID
echo "Job ${SLURM_ARRAY_TASK_ID} starting at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Temp: $TEMP_DIR"

# Run preprocessing for this shard
python download_and_preprocess_vggsound.py \
  --csv_path=PreProcessing/vggsound_train.csv \
  --output_path=$OUTPUT_DIR \
  --num_shards=100 \
  --temp_dir=$TEMP_DIR \
  --check_duration=True \
  --save_progress_every=10

# Cleanup temp directory
rm -rf $TEMP_DIR

echo "Job ${SLURM_ARRAY_TASK_ID} finished at $(date)"
