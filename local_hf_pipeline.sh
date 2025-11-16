#!/bin/bash

# LOCAL VGGSound Pipeline - Process everything on local machine
# Downloads from HuggingFace, processes locally with GPU, saves to large storage
# 
# Process:
# 1. Download ONE tar file from HuggingFace (~17GB)
# 2. Extract videos to temporary directory
# 3. Process videos into TFRecords using local GPU
# 4. Save TFRecords to large storage
# 5. Cleanup (delete tar + videos)
# 6. Repeat for next tar

set -e  # Exit on any error

# Configuration
START_TAR=${1:-0}  # Which tar file to start from (0-19)
BATCH_SIZE=3000
NUM_TARS=20
NUM_WORKERS=24  # Number of parallel workers for video processing

# Paths
WORK_DIR="$(pwd)"
TEMP_DIR="${WORK_DIR}/vggsound_temp"
STORAGE_BASE="/media/labuta/7f1ad7d2-a1d3-4a1f-ae81-7cb5dd2661a3/VGG_Preprocessed"

# HuggingFace dataset URL
BASE_URL="https://huggingface.co/datasets/Loie/VGGSound/resolve/main"

echo "============================================"
echo "VGGSound LOCAL Pipeline - TRAIN & TEST"
echo "============================================"
echo "Batch size: $BATCH_SIZE"
echo "Starting from tar: $START_TAR"
echo "Temporary directory: $TEMP_DIR"
echo ""

# Check if we have enough space
REQUIRED_SPACE_GB=20
AVAILABLE_SPACE_GB=$(df -BG "$WORK_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    echo "WARNING: Low disk space. Available: ${AVAILABLE_SPACE_GB}GB, Recommended: ${REQUIRED_SPACE_GB}GB"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "scenic_preprocessing" ]; then
    echo "ERROR: Please activate the scenic_preprocessing conda environment first:"
    echo "  conda activate scenic_preprocessing"
    exit 1
fi

# Check if Python packages are available
python -c "import scenic" 2>/dev/null || {
    echo "ERROR: scenic package not found. Please ensure scenic_phd environment is properly set up."
    exit 1
}

# Check all required dependencies
echo "Checking dependencies..."
python -c "
import sys
missing = []
try:
    import pandas
except ImportError:
    missing.append('pandas')
try:
    import ffmpeg
except ImportError:
    missing.append('ffmpeg-python')
try:
    import librosa
except ImportError:
    missing.append('librosa')
try:
    import tensorflow
except ImportError:
    missing.append('tensorflow')

if missing:
    print(f'ERROR: Missing required packages: {missing}')
    print('Please install with: pip install ' + ' '.join(missing))
    sys.exit(1)
print('✓ All dependencies installed')
" || exit 1

# Create directories
mkdir -p "$TEMP_DIR"

echo "✓ Environment checks passed"
echo ""

# Process each tar file one by one
for ((tar_id=START_TAR; tar_id<NUM_TARS; tar_id++)); do
    TAR_NUM=$(printf "%02d" $tar_id)
    TAR_FILE="vggsound_${TAR_NUM}.tar.gz"
    TAR_PATH="${TEMP_DIR}/${TAR_FILE}"
    
    echo ""
    echo "========================================"
    echo "Processing TAR $tar_id/$((NUM_TARS-1)): $TAR_FILE"
    echo "========================================"
    
    VIDEO_DIR="${TEMP_DIR}/videos_${TAR_NUM}"
    
    VIDEO_DIR="${TEMP_DIR}/videos_${TAR_NUM}"
    
    # Check if videos are already extracted
    VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    
    if [ "$VIDEO_COUNT" -gt 0 ]; then
        echo "✓ Found $VIDEO_COUNT videos already extracted in $VIDEO_DIR"
        echo "Skipping download and extraction steps."
        echo ""
    else
        # Step 1: Download tar file
        echo "=== STEP 1: Downloading tar file from HuggingFace ==="
        
        if [ -f "$TAR_PATH" ]; then
            echo "Tar file already exists, checking integrity..."
            if gzip -t "$TAR_PATH" 2>/dev/null; then
                echo "✓ Tar file integrity verified!"
            else
                echo "⚠ Tar file corrupted, re-downloading..."
                rm -f "$TAR_PATH"
                rm -f "${TAR_PATH}.aria2"  # Remove aria2 control file
            fi
        fi
        
        if [ ! -f "$TAR_PATH" ]; then
            echo "Downloading ${TAR_FILE} (~17GB, this may take 10-30 minutes)..."
            aria2c \
                --max-connection-per-server=16 \
                --split=16 \
                --min-split-size=1M \
                --max-tries=5 \
                --retry-wait=5 \
                --continue=true \
                --console-log-level=notice \
                --summary-interval=10 \
                "${BASE_URL}/${TAR_FILE}" \
                --dir="$(dirname "$TAR_PATH")" \
                --out="$(basename "$TAR_PATH")" || {
                echo "ERROR: Download failed!"
                exit 1
            }
            echo "✓ Download complete!"
        fi
        
        ls -lh "$TAR_PATH"
        echo ""
        
        # Step 2: Extract videos
        echo "=== STEP 2: Extracting videos ==="
        
        mkdir -p "$VIDEO_DIR"
        
        echo "Extracting to: $VIDEO_DIR"
        tar -xzf "$TAR_PATH" -C "$VIDEO_DIR" --strip-components=7 2>&1 | tail -20 || true
        
        # Count extracted videos
        VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
        
        echo "✓ Extracted $VIDEO_COUNT videos"
        
        # Delete tar to save space
        echo "Deleting tar file to save space..."
        rm -f "$TAR_PATH"
        echo ""
    fi
    
    # Final check: ensure we have videos to process
    if [ "$VIDEO_COUNT" -eq 0 ]; then
        echo "ERROR: No videos found in $VIDEO_DIR!"
        rm -rf "$VIDEO_DIR"
        continue
    fi
    
    # Process both train and test splits from the same extracted videos
    for SPLIT in train test; do
        echo ""
        echo "###############################################"
        echo "# Processing ${SPLIT^^} split from TAR $tar_id"
        echo "###############################################"
        echo ""
        
        OUTPUT_DIR="${STORAGE_BASE}/${SPLIT}_tfrecords_local"
        CSV_FILE="${WORK_DIR}/PreProcessing/vggsound_${SPLIT}.csv"
        
        # Count total videos in CSV (excluding header)
        if [ ! -f "$CSV_FILE" ]; then
            echo "ERROR: CSV file not found: $CSV_FILE"
            echo "Skipping ${SPLIT} split..."
            continue
        fi
        
        TOTAL_VIDEOS=$(tail -n +2 "$CSV_FILE" | wc -l)
        
        echo "Split: ${SPLIT}"
        echo "Total videos in CSV: $TOTAL_VIDEOS"
        echo "Output directory: $OUTPUT_DIR"
        echo "CSV file: $CSV_FILE"
        echo ""
        
        # Create output directory
        mkdir -p "$OUTPUT_DIR"
        
        # Step 3: Process videos into TFRecords
        echo "=== STEP 3: Processing videos into TFRecords for ${SPLIT} ==="
        
        # Calculate which batches contain videos from this tar
        TOTAL_BATCHES=$(( (TOTAL_VIDEOS + BATCH_SIZE - 1) / BATCH_SIZE ))
        
        echo "Processing in batches of $BATCH_SIZE videos..."
        echo "Total batches: $TOTAL_BATCHES"
        echo ""
        
        # Process each batch
        for ((batch=0; batch<TOTAL_BATCHES; batch++)); do
            start_row=$((batch * BATCH_SIZE))
            end_row=$((start_row + BATCH_SIZE))
            
            if [ $end_row -gt $TOTAL_VIDEOS ]; then
                end_row=$TOTAL_VIDEOS
            fi
            
            # Include tar number in batch directory name to avoid conflicts
            batch_id=$(printf "tar%02d_batch%03d" $tar_id $batch)
            batch_output_dir="${OUTPUT_DIR}/${batch_id}"
            
            # Skip if batch already processed
            if [ -f "${batch_output_dir}/.complete" ]; then
                echo "  Batch $batch (rows $start_row-$end_row): Already complete, skipping"
                continue
            fi
            
            mkdir -p "$batch_output_dir"
            
            echo "  Processing batch $batch/$((TOTAL_BATCHES-1)) (rows $start_row-$end_row)..."
            
            # Run the Python preprocessing script with parallel workers
            python PreProcessing/preprocess_vggsound_local.py \
                --csv_file "$CSV_FILE" \
                --video_dir "$VIDEO_DIR" \
                --output_dir "$batch_output_dir" \
                --start_row $start_row \
                --end_row $end_row \
                --split $SPLIT \
                --num_workers $NUM_WORKERS 2>&1 | grep -E "Processing|Using|Created|ERROR|Warning|Processed" | tail -10 || true
            
            # Mark as complete if any tfrecords were created
            if [ -n "$(find "$batch_output_dir" -name "*.tfrecord" 2>/dev/null)" ]; then
                touch "${batch_output_dir}/.complete"
                tfrecord_count=$(find "$batch_output_dir" -name "*.tfrecord" 2>/dev/null | wc -l)
                echo "    ✓ Created $tfrecord_count TFRecord file(s)"
            else
                echo "    (no videos from this batch in current tar)"
            fi
        done
        
        echo ""
        echo "✓ Completed ${SPLIT} processing for tar $tar_id/$((NUM_TARS-1))"
        
        # Show progress for this split
        completed_batches=$(find "$OUTPUT_DIR" -name ".complete" 2>/dev/null | wc -l)
        echo "Overall progress for ${SPLIT}: $completed_batches/$TOTAL_BATCHES batches completed"
        echo ""
        
    done  # End of split loop (train/test)
    
    # Cleanup after processing both splits
    echo ""
    echo "=== STEP 4: Cleanup ==="
    echo "Deleting extracted videos to save space..."
    rm -rf "$VIDEO_DIR"
    
    # Show storage status
    echo ""
    echo "Storage status:"
    df -h "$WORK_DIR" | tail -1
    echo ""
    
    echo "✓ Completed all splits for tar $tar_id/$((NUM_TARS-1))"
    echo ""

done  # End of tar loop

# Final summary
echo ""
echo "============================================"
echo "Pipeline Complete - ALL SPLITS!"
echo "============================================"

# Summary for both splits
for SPLIT in train test; do
    OUTPUT_DIR="${STORAGE_BASE}/${SPLIT}_tfrecords_local"
    if [ -d "$OUTPUT_DIR" ]; then
        completed_batches=$(find "$OUTPUT_DIR" -name ".complete" 2>/dev/null | wc -l)
        total_tfrecords=$(find "$OUTPUT_DIR" -name "*.tfrecord" 2>/dev/null | wc -l)
        
        echo ""
        echo "${SPLIT^^} split:"
        echo "  Output: $OUTPUT_DIR"
        echo "  Completed batches: $completed_batches"
        echo "  Total TFRecord files: $total_tfrecords"
    fi
done

echo ""

# Cleanup temp directory
echo "Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"

echo "✓ All done!"
