#!/bin/bash

# Automated pipeline to download VGGSound from HuggingFace and preprocess into TFRecords
# LOW STORAGE MODE: Process one tar file at a time to minimize cluster storage usage
# 1. Download ONE tar file
# 2. Extract it
# 3. Process videos from that tar into TFRecords
# 4. Download TFRecords to local machine
# 5. Cleanup cluster (delete tar, extracted videos, TFRecords)
# 6. Repeat for next tar

set -e  # Exit on any error

# Configuration
CLUSTER_USER="bravhee"
CLUSTER_HOST="mentat001.dccn.nl"
CLUSTER_PATH="scenic_PhD"
SPLIT=${1:-train}  # train or test
START_TAR=${2:-0}  # Which tar file to start from (0-19)

# Get total videos for the split
if [ "$SPLIT" == "train" ]; then
    TOTAL_VIDEOS=170752  # Training videos
    CSV_NAME="vggsound_train.csv"
else
    TOTAL_VIDEOS=15122   # Test videos
    CSV_NAME="vggsound_test.csv"
fi

BATCH_SIZE=200
PARALLEL_JOBS=3  # Process 3 batches in parallel per tar file
NUM_TARS=20      # Total number of tar files

echo "============================================"
echo "VGGSound HuggingFace Pipeline - ${SPLIT^^} SET"
echo "LOW STORAGE MODE (one tar at a time)"
echo "============================================"
echo "Total videos: $TOTAL_VIDEOS"
echo "Batch size: $BATCH_SIZE"
echo "Parallel jobs per tar: $PARALLEL_JOBS"
echo "Starting from tar: $START_TAR"
echo ""

# Create local output directory
mkdir -p "${SPLIT}_tfrecords_local"

# Process each tar file one by one
for ((tar_id=START_TAR; tar_id<NUM_TARS; tar_id++)); do
    TAR_NUM=$(printf "%02d" $tar_id)
    TAR_FILE="vggsound_${TAR_NUM}.tar.gz"
    BASE_URL="https://huggingface.co/datasets/Loie/VGGSound/resolve/main"
    
    echo ""
    echo "========================================"
    echo "Processing TAR $tar_id/$((NUM_TARS-1)): $TAR_FILE"
    echo "========================================"
    
    # Step 1: Download tar file on cluster
    echo "=== STEP 1: Downloading tar file on cluster ==="
    ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} << EOF
        cd ${CLUSTER_PATH}
        mkdir -p vggsound_data
        cd vggsound_data
        
        echo "Checking cluster storage..."
        df -h ~ | tail -1
        
        if [ -f "${TAR_FILE}" ]; then
            echo "Tar file already exists, skipping download"
        else
            echo "Downloading ${TAR_FILE} (~17GB, ~30-60 min)..."
            wget -c "${BASE_URL}/${TAR_FILE}" -O "${TAR_FILE}" || exit 1
            echo "Download complete!"
        fi
        
        ls -lh "${TAR_FILE}"
EOF
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download tar file!"
        exit 1
    fi
    
    # Step 2: Extract tar file
    echo ""
    echo "=== STEP 2: Extracting tar file ==="
    ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} << EOF
        cd ${CLUSTER_PATH}/vggsound_data
        
        echo "Extracting ${TAR_FILE}..."
        mkdir -p videos
        tar -xzf "${TAR_FILE}" -C videos --strip-components=6 || exit 1
        echo "Extraction complete!"
        
        echo "Extracted videos:"
        ls videos | head -20
        echo "Total videos in this tar:"
        ls videos | wc -l
        
        echo "Cluster storage after extraction:"
        df -h ~ | tail -1
EOF
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract tar file!"
        exit 1
    fi
    
    # Step 3: Calculate which batches use videos from this tar
    # We need to check which videos are in this tar and process only those batches
    echo ""
    echo "=== STEP 3: Identifying batches to process from this tar ==="
    
    # Get list of video files in this tar
    VIDEO_LIST=$(ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} \
        "ls ${CLUSTER_PATH}/vggsound_data/videos/*.mp4 2>/dev/null | xargs -n1 basename" | tr '\n' '|' | sed 's/|$//')
    
    if [ -z "$VIDEO_LIST" ]; then
        echo "WARNING: No videos found in tar $tar_id, skipping..."
        # Cleanup and continue
        ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} \
            "rm -rf ${CLUSTER_PATH}/vggsound_data/${TAR_FILE} ${CLUSTER_PATH}/vggsound_data/videos/*"
        continue
    fi
    
    echo "Found $(echo $VIDEO_LIST | tr '|' '\n' | wc -l) videos in this tar"
    
    # Step 4: Process all CSV rows, but only videos from this tar will succeed
    echo ""
    echo "=== STEP 4: Processing batches (only videos from this tar) ==="
    
    TOTAL_BATCHES=$(( (TOTAL_VIDEOS + BATCH_SIZE - 1) / BATCH_SIZE ))
    
    # Process in chunks of PARALLEL_JOBS batches
    for ((batch=0; batch<TOTAL_BATCHES; batch+=PARALLEL_JOBS)); do
        echo ""
        echo "--- Submitting batch chunk starting at $batch ---"
        
        # Submit parallel jobs for this chunk
        job_ids=()
        for ((i=0; i<PARALLEL_JOBS; i++)); do
            current_batch=$((batch + i))
            if [ $current_batch -ge $TOTAL_BATCHES ]; then
                break
            fi
            
            start_row=$((current_batch * BATCH_SIZE))
            end_row=$((start_row + BATCH_SIZE))
            
            if [ $end_row -gt $TOTAL_VIDEOS ]; then
                end_row=$TOTAL_VIDEOS
            fi
            
            if [ $start_row -ge $TOTAL_VIDEOS ]; then
                break
            fi
            
            echo "  Submitting batch $current_batch: rows $start_row-$end_row"
            job_id=$(ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} \
                "cd ${CLUSTER_PATH} && sbatch slurm_process_local.sh $start_row $end_row $SPLIT" | awk '{print $NF}')
            job_ids+=($job_id)
        done
        
        # Wait for these jobs to complete
        if [ ${#job_ids[@]} -gt 0 ]; then
            echo "Waiting for ${#job_ids[@]} jobs to complete..."
            while true; do
                job_count=$(ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} \
                    "squeue -u ${CLUSTER_USER} -n vgg_process_local -h | wc -l")
                
                if [ "$job_count" -eq "0" ]; then
                    echo "Batch chunk complete!"
                    break
                fi
                
                echo "  $job_count jobs still running..."
                sleep 30
            done
            
            # Download TFRecord archives that were created
            echo "Downloading TFRecord archives..."
            for ((i=0; i<PARALLEL_JOBS; i++)); do
                current_batch=$((batch + i))
                if [ $current_batch -ge $TOTAL_BATCHES ]; then
                    break
                fi
                
                start_row=$((current_batch * BATCH_SIZE))
                batch_id=$(printf "%05d" $start_row)
                archive="${SPLIT}_batch_${batch_id}.tar.gz"
                
                # Try to download (may not exist if no videos from this tar were in the batch)
                scp -o LogLevel=ERROR \
                    ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}/PreProcessing/${archive} \
                    ${SPLIT}_tfrecords_local/ 2>/dev/null && echo "  Downloaded $archive" || echo "  (no archive for batch $current_batch)"
                
                # Delete remote archive
                ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} \
                    "rm -f ${CLUSTER_PATH}/PreProcessing/${archive}" 2>/dev/null || true
            done
        fi
    done
    
    # Step 5: Cleanup cluster storage before next tar
    echo ""
    echo "=== STEP 5: Cleaning up cluster storage ==="
    ssh -o LogLevel=ERROR ${CLUSTER_USER}@${CLUSTER_HOST} << EOF
        cd ${CLUSTER_PATH}
        
        echo "Deleting tar file and extracted videos..."
        rm -f vggsound_data/${TAR_FILE}
        rm -rf vggsound_data/videos/*
        rm -f PreProcessing/tfrecords_${SPLIT}_local/batch_*/*.tfrecord
        
        echo "Cluster storage after cleanup:"
        df -h ~ | tail -1
EOF
    
    echo ""
    echo "✓ Completed tar $tar_id/$((NUM_TARS-1))"
    echo "  TFRecords downloaded to: ${SPLIT}_tfrecords_local/"
    echo ""
done

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo "All TFRecord archives are in: ${SPLIT}_tfrecords_local/"
ls -lh ${SPLIT}_tfrecords_local/ | head -30
echo ""
echo "Total archives downloaded:"
ls ${SPLIT}_tfrecords_local/*.tar.gz | wc -l
echo ""
echo "Extract with: tar -xzf ${SPLIT}_tfrecords_local/${SPLIT}_batch_XXXXX.tar.gz"
