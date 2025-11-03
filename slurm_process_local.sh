#!/bin/bash
#SBATCH --job-name=vgg_process_local
#SBATCH --output=/home/mpla/bravhee/scenic_PhD/logs/process_local_%A.out
#SBATCH --error=/home/mpla/bravhee/scenic_PhD/logs/process_local_%A.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Process local VGGSound videos into TFRecords
# This assumes videos have been extracted from HuggingFace tar files

START_ROW=${1:-0}
END_ROW=${2:-200}
BATCH_ID=$(printf "%05d" $START_ROW)
SPLIT=${3:-train}  # train or test

# Activate conda environment
module load anaconda3
source activate scenic_preprocessing

export KMP_DUPLICATE_LIB_OK=TRUE

LOCAL_VIDEOS_DIR="$HOME/scenic_PhD/vggsound_data/video"
OUTPUT_DIR="$HOME/scenic_PhD/PreProcessing/tfrecords_${SPLIT}_local/batch_${BATCH_ID}"
TEMP_DIR="$HOME/scenic_PhD/PreProcessing/temp_${SPLIT}_local"

mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR
mkdir -p $HOME/scenic_PhD/logs

echo "=== Processing ${SPLIT^^} Videos $START_ROW to $END_ROW from local files ==="
echo "Local videos: $LOCAL_VIDEOS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Space check:"
df -h $HOME | tail -1

cd $HOME/scenic_PhD

# Select CSV based on split
if [ "$SPLIT" == "train" ]; then
    CSV_FILE="Video_csvs/vggsound_train.csv"
else
    CSV_FILE="Video_csvs/vggsound_test.csv"
fi

# Create a temporary CSV with just these rows
# Row numbers are 0-indexed (row 0 is first data row after header)
# We need to skip the header (line 1), then get START_ROW to END_ROW
head -n 1 $CSV_FILE > /tmp/vggsound_${SPLIT}_batch_${BATCH_ID}.csv
tail -n +2 $CSV_FILE | head -n $END_ROW | tail -n $((END_ROW - START_ROW)) >> /tmp/vggsound_${SPLIT}_batch_${BATCH_ID}.csv

echo "Processing $(wc -l < /tmp/vggsound_${SPLIT}_batch_${BATCH_ID}.csv) rows (including header)"

# Process using local videos
python download_and_preprocess_vggsound.py \
  --csv_path=/tmp/vggsound_${SPLIT}_batch_${BATCH_ID}.csv \
  --output_path=$OUTPUT_DIR \
  --temp_dir=$TEMP_DIR \
  --num_shards=2 \
  --target_fps=25 \
  --decode_audio=True \
  --skip_existing=False \
  --local_videos_dir=$LOCAL_VIDEOS_DIR \
  --require_local=True \
  --local_are_clips=True \
  --check_duration=False

if [ $? -ne 0 ]; then
    echo "ERROR: Processing failed"
    exit 1
fi

echo "Processing complete!"
echo "Output directory:"
ls -lh $OUTPUT_DIR

# Create tar archive of TFRecords
ARCHIVE_NAME="${SPLIT}_batch_${BATCH_ID}.tar.gz"
cd $HOME/scenic_PhD/PreProcessing
tar -czf $ARCHIVE_NAME tfrecords_${SPLIT}_local/batch_${BATCH_ID}/*.tfrecord

if [ -f "$ARCHIVE_NAME" ]; then
    echo "Archive created: $ARCHIVE_NAME"
    ls -lh $ARCHIVE_NAME
    echo "$HOME/scenic_PhD/PreProcessing/$ARCHIVE_NAME"
else
    echo "ERROR: Failed to create archive"
    exit 1
fi

# Cleanup temp files
rm -f /tmp/vggsound_${SPLIT}_batch_${BATCH_ID}.csv
rm -f /tmp/vggsound_${SPLIT}_batch_${BATCH_ID}_data.csv

echo "=== Completed batch ${BATCH_ID} for ${SPLIT} ==="
