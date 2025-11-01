#!/bin/bash
#SBATCH --job-name=vggsound_test_micro
#SBATCH --output=/home/mpla/bravhee/scenic_PhD/logs/test_micro_%A.out
#SBATCH --error=/home/mpla/bravhee/scenic_PhD/logs/test_micro_%A.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Process test set in small batches
START_ROW=${1:-0}
END_ROW=${2:-100}
BATCH_ID=$(printf "%05d" $START_ROW)

# Activate conda environment
module load anaconda3
source activate scenic_preprocessing

export KMP_DUPLICATE_LIB_OK=TRUE

OUTPUT_DIR="$HOME/scenic_PhD/PreProcessing/tfrecords_test_micro/batch_${BATCH_ID}"
TEMP_DIR="$HOME/scenic_PhD/PreProcessing/temp_test_micro"

mkdir -p $OUTPUT_DIR
mkdir -p $TEMP_DIR
mkdir -p $HOME/scenic_PhD/logs

echo "=== Processing TEST Videos $START_ROW to $END_ROW ==="
echo "Output: $OUTPUT_DIR"
echo "Space check:"
df -h $HOME | tail -1

cd $HOME/scenic_PhD

# Create a temporary CSV with just these rows
# Row numbers are 0-indexed (row 0 is first data row after header)
# Add 1 for header, add 1 for inclusive end row
HEADER_OFFSET=1
head -n $((END_ROW + HEADER_OFFSET + 1)) PreProcessing/vggsound_test.csv | tail -n $((END_ROW - START_ROW)) > /tmp/vggsound_test_micro_${BATCH_ID}.csv

# Add header
head -n 1 PreProcessing/vggsound_test.csv > /tmp/vggsound_test_batch_${BATCH_ID}.csv
cat /tmp/vggsound_test_micro_${BATCH_ID}.csv >> /tmp/vggsound_test_batch_${BATCH_ID}.csv

# Process this small batch
# Use cookies file if it exists, otherwise try browser cookies
if [ -f "$HOME/scenic_PhD/youtube_cookies.txt" ]; then
  COOKIE_ARG="--cookies_file=$HOME/scenic_PhD/youtube_cookies.txt"
else
  COOKIE_ARG="--cookies_from_browser=chrome"
fi

python download_and_preprocess_vggsound.py \
  --csv_path=/tmp/vggsound_test_batch_${BATCH_ID}.csv \
  --output_path=$OUTPUT_DIR \
  --num_shards=1 \
  --temp_dir=$TEMP_DIR \
  --check_duration=True \
  --save_progress_every=10 \
  $COOKIE_ARG

# Archive immediately
cd $OUTPUT_DIR
tar -czf $HOME/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz *.tfrecord 2>/dev/null

echo ""
echo "=== Test Batch Complete ==="
echo "Archive: ~/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz"
ls -lh $HOME/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz 2>/dev/null
echo ""
echo "Download with:"
echo "  scp bravhee@mentat001:~/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz ."
echo ""
echo "After downloading, DELETE:"
echo "  ssh bravhee@mentat001 'rm ~/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz && rm -rf $OUTPUT_DIR'"

# Cleanup
rm -rf $TEMP_DIR
rm /tmp/vggsound_test_micro_${BATCH_ID}.csv
rm /tmp/vggsound_test_batch_${BATCH_ID}.csv

echo "Finished at $(date)"
