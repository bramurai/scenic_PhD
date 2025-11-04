#!/bin/bash
# Bash script to extract all TFRecord tar archives
# Maintains the batch_* directory structure that the model expects

SPLIT=${1:-train}  # "train" or "test"
SOURCE_DIR="${SPLIT}_tfrecords_local"
TARGET_DIR="${SPLIT}_tfrecords_local"

echo "============================================"
echo "Extracting TFRecord archives for: $SPLIT"
echo "============================================"

# Count total archives
TOTAL_ARCHIVES=$(ls ${SOURCE_DIR}/*.tar.gz 2>/dev/null | wc -l)
echo "Found $TOTAL_ARCHIVES tar archives to extract"
echo ""

EXTRACTED_COUNT=0
SKIPPED_COUNT=0

for archive in ${SOURCE_DIR}/*.tar.gz; do
    if [ ! -f "$archive" ]; then
        continue
    fi
    
    archive_name=$(basename "$archive")
    
    # Extract batch ID from filename (e.g., train_tar00_batch_00000.tar.gz -> batch_00000)
    if [[ $archive_name =~ batch_([0-9]+)\.tar\.gz$ ]]; then
        batch_id="${BASH_REMATCH[1]}"
        batch_dir="batch_${batch_id}"
        target_path="${TARGET_DIR}/${batch_dir}"
        
        # Check if already extracted
        if [ -d "$target_path" ]; then
            tfrecord_count=$(ls ${target_path}/*.tfrecord 2>/dev/null | wc -l)
            if [ "$tfrecord_count" -gt 0 ]; then
                echo "  [SKIP] $archive_name -> $batch_dir (already extracted: $tfrecord_count files)"
                ((SKIPPED_COUNT++))
                continue
            fi
        fi
        
        echo "  [EXTRACT] $archive_name -> $batch_dir"
        
        # Extract
        if tar -xzf "$archive" -C "$TARGET_DIR" 2>/dev/null; then
            tfrecord_count=$(ls ${target_path}/*.tfrecord 2>/dev/null | wc -l)
            echo "    ✓ Extracted $tfrecord_count TFRecord files"
            ((EXTRACTED_COUNT++))
        else
            echo "    ✗ Extraction failed!"
        fi
    else
        echo "  [SKIP] $archive_name (unexpected filename format)"
        ((SKIPPED_COUNT++))
    fi
done

echo ""
echo "============================================"
echo "Extraction Complete!"
echo "============================================"
echo "Extracted: $EXTRACTED_COUNT archives"
echo "Skipped: $SKIPPED_COUNT archives (already extracted)"
echo ""

# Count total TFRecord files
TOTAL_TFRECORDS=$(find ${TARGET_DIR} -name "*.tfrecord" 2>/dev/null | wc -l)
TOTAL_BATCHES=$(find ${TARGET_DIR} -maxdepth 1 -type d -name "batch_*" 2>/dev/null | wc -l)

echo "Total batch directories: $TOTAL_BATCHES"
echo "Total TFRecord files: $TOTAL_TFRECORDS"
echo ""
echo "Dataset ready at: $TARGET_DIR"
echo ""
echo "You can now delete the .tar.gz files to save space:"
echo "  rm ${SOURCE_DIR}/*.tar.gz"
