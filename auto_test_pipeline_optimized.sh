#!/bin/bash
# CHUNKED PROCESSING: Process 25 batches at a time, wait for completion, download, repeat - TEST

TOTAL_VIDEOS=15122  # Test set size
BATCH_SIZE=200  # Process 200 videos at a time (~1.1GB each)
PARALLEL_JOBS=10  # Run 10 jobs simultaneously (reduced to avoid rate limiting)
PARALLEL_DOWNLOADS=25  # Download 25 batches simultaneously (1 Gbps connection can handle it!)
LAPTOP_DIR="$1"

if [ -z "$LAPTOP_DIR" ]; then
    echo "Usage: bash auto_test_pipeline_optimized.sh <laptop_download_dir>"
    exit 1
fi

mkdir -p "$LAPTOP_DIR"

# Generate all batch starting points
ALL_STARTS=($(seq 0 $BATCH_SIZE $((TOTAL_VIDEOS - 1))))
TOTAL_BATCHES=${#ALL_STARTS[@]}

echo "=== VGGSound TEST Pipeline (CHUNKED) ==="
echo "Total videos: $TOTAL_VIDEOS"
echo "Batch size: $BATCH_SIZE videos"
echo "Total batches: $TOTAL_BATCHES"
echo "Processing chunks of: $PARALLEL_JOBS batches"
echo ""

# Clean up old runs
echo "=== Cleaning up old runs on cluster ==="
ssh -o LogLevel=ERROR bravhee@mentat001.dccn.nl "rm -rf ~/scenic_PhD/PreProcessing/tfrecords_test_micro/* ~/scenic_PhD/PreProcessing/test_batch_*.tar.gz" 2>/dev/null
echo "✓ Cleanup complete"
echo ""

TOTAL_DOWNLOADED=0
TOTAL_FAILED=0

# Process in chunks of PARALLEL_JOBS
for ((CHUNK_START=0; CHUNK_START<TOTAL_BATCHES; CHUNK_START+=PARALLEL_JOBS)); do
    CHUNK_END=$((CHUNK_START + PARALLEL_JOBS))
    if [ $CHUNK_END -gt $TOTAL_BATCHES ]; then
        CHUNK_END=$TOTAL_BATCHES
    fi
    
    CHUNK_NUM=$((CHUNK_START / PARALLEL_JOBS + 1))
    TOTAL_CHUNKS=$(((TOTAL_BATCHES + PARALLEL_JOBS - 1) / PARALLEL_JOBS))
    
    echo "=========================================="
    echo "=== CHUNK $CHUNK_NUM of $TOTAL_CHUNKS ==="
    echo "=========================================="
    echo ""
    
    # Arrays for this chunk
    declare -a CHUNK_BATCH_IDS
    
    # Submit batches for this chunk
    echo "=== Submitting batches $(($CHUNK_START + 1)) to $CHUNK_END ==="
    for ((i=CHUNK_START; i<CHUNK_END; i++)); do
        START=${ALL_STARTS[$i]}
        END=$((START + BATCH_SIZE))
        BATCH_ID=$(printf "%05d" $START)
        
        ssh -o LogLevel=ERROR bravhee@mentat001.dccn.nl "cd scenic_PhD && sbatch slurm_test_micro.sh $START $END" > /dev/null 2>&1
        CHUNK_BATCH_IDS+=($BATCH_ID)
        echo "[$(date +%H:%M:%S)] ✓ Submitted batch $BATCH_ID ($START-$END)"
        sleep 1
    done
    
    echo ""
    echo "=== Waiting for chunk to complete ==="
    
    # Wait for all jobs in this chunk to complete
    while true; do
        RUNNING=$(ssh -o LogLevel=ERROR bravhee@mentat001.dccn.nl "squeue -u bravhee -n vggsound_test_micro -h | wc -l" 2>/dev/null || echo "0")
        
        if [ $RUNNING -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] All jobs completed!"
            break
        fi
        
        echo "[$(date +%H:%M:%S)] Waiting... ($RUNNING jobs still running)"
        sleep 15
    done
    
    echo ""
    echo "=== Downloading completed batches ==="
    
    # Download all batches from this chunk in parallel
    for BATCH_ID in "${CHUNK_BATCH_IDS[@]}"; do
        (
            ARCHIVE="test_batch_${BATCH_ID}.tar.gz"
            
            if ssh -o LogLevel=ERROR bravhee@mentat001.dccn.nl "test -f ~/scenic_PhD/PreProcessing/$ARCHIVE" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] Downloading $ARCHIVE..."
                
                if scp -o LogLevel=ERROR -q bravhee@mentat001.dccn.nl:~/scenic_PhD/PreProcessing/$ARCHIVE "$LAPTOP_DIR/" 2>/dev/null; then
                    echo "[$(date +%H:%M:%S)] ✓ Downloaded: $ARCHIVE"
                    
                    # Delete from cluster
                    ssh -o LogLevel=ERROR bravhee@mentat001.dccn.nl "rm ~/scenic_PhD/PreProcessing/$ARCHIVE; rm -rf ~/scenic_PhD/PreProcessing/tfrecords_test_micro/batch_${BATCH_ID}" 2>/dev/null
                    echo "[$(date +%H:%M:%S)] ✓ Cleaned up cluster: batch_${BATCH_ID}"
                else
                    echo "[$(date +%H:%M:%S)] ✗ Download failed: $ARCHIVE"
                fi
            else
                echo "[$(date +%H:%M:%S)] ✗ Batch $BATCH_ID failed (no archive created)"
            fi
        ) &
    done
    
    # Wait for all downloads to complete
    wait
    
    # Count successes/failures for this chunk
    CHUNK_DOWNLOADED=0
    CHUNK_FAILED=0
    for BATCH_ID in "${CHUNK_BATCH_IDS[@]}"; do
        if [ -f "$LAPTOP_DIR/test_batch_${BATCH_ID}.tar.gz" ]; then
            CHUNK_DOWNLOADED=$((CHUNK_DOWNLOADED + 1))
        else
            CHUNK_FAILED=$((CHUNK_FAILED + 1))
        fi
    done
    
    TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + CHUNK_DOWNLOADED))
    TOTAL_FAILED=$((TOTAL_FAILED + CHUNK_FAILED))
    
    echo ""
    echo "=== Chunk $CHUNK_NUM Complete ==="
    echo "Downloaded: $CHUNK_DOWNLOADED batches"
    echo "Failed: $CHUNK_FAILED batches"
    echo "Total progress: $TOTAL_DOWNLOADED/$TOTAL_BATCHES downloaded"
    echo ""
    
    unset CHUNK_BATCH_IDS
done

echo ""
echo "=========================================="
echo "=== TEST Pipeline Complete! ==="
echo "=========================================="
echo "Total downloaded: $TOTAL_DOWNLOADED batches"
echo "Total failed: $TOTAL_FAILED batches"
echo "Location: $LAPTOP_DIR"
