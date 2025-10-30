#!/bin/bash
# OPTIMIZED: Download batches immediately as they complete (parallel download)

TOTAL_VIDEOS=600 # 15122  # Test set size
BATCH_SIZE=50  # Process 200 videos at a time (~1.1GB each)
PARALLEL_JOBS=25  # Run 25 jobs simultaneously
PARALLEL_DOWNLOADS=25  # Download 25 batches simultaneously (1 Gbps connection can handle it!)
LAPTOP_DIR="$1"

if [ -z "$LAPTOP_DIR" ]; then
    echo "Usage: bash auto_test_pipeline_optimized.sh <laptop_download_dir>"
    exit 1
fi

mkdir -p "$LAPTOP_DIR"

echo "=== VGGSound TEST Pipeline (OPTIMIZED) ==="
echo "Total videos: $TOTAL_VIDEOS"
echo "Batch size: $BATCH_SIZE videos"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Parallel downloads: $PARALLEL_DOWNLOADS"
echo "Total batches: $((TOTAL_VIDEOS / BATCH_SIZE))"
echo ""

# Track submitted batches
declare -A BATCH_STATUS  # batch_id -> "submitted" | "completed" | "downloaded"
declare -a ALL_BATCHES

# Function to download a batch in background
download_batch() {
    local BATCH_ID=$1
    local ARCHIVE="test_batch_${BATCH_ID}.tar.gz"
    
    echo "[$(date +%H:%M:%S)] Downloading $ARCHIVE..."
    
    if scp -q bravhee@mentat001:~/scenic_PhD/PreProcessing/$ARCHIVE "$LAPTOP_DIR/" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ✓ Downloaded: $ARCHIVE"
        
        # Delete from cluster
        ssh bravhee@mentat001 "rm ~/scenic_PhD/PreProcessing/$ARCHIVE; rm -rf ~/scenic_PhD/PreProcessing/tfrecords_test_micro/batch_${BATCH_ID}" 2>/dev/null
        echo "[$(date +%H:%M:%S)] ✓ Cleaned up cluster: batch_${BATCH_ID}"
        
        BATCH_STATUS[$BATCH_ID]="downloaded"
    else
        echo "[$(date +%H:%M:%S)] ✗ Download failed: $ARCHIVE"
        BATCH_STATUS[$BATCH_ID]="failed"
    fi
}

# Submit all batches
echo "=== Submitting batches ==="
for START in $(seq 0 $BATCH_SIZE $((TOTAL_VIDEOS - BATCH_SIZE))); do
    END=$((START + BATCH_SIZE))
    BATCH_ID=$(printf "%05d" $START)
    
    # Wait if too many jobs running
    RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l")
    while [ $RUNNING -ge $PARALLEL_JOBS ]; do
        sleep 10
        RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l")
    done
    
    ssh bravhee@mentat001 "cd scenic_PhD && sbatch slurm_test_micro.sh $START $END" > /dev/null 2>&1
    sleep 1
    
    BATCH_STATUS[$BATCH_ID]="submitted"
    ALL_BATCHES+=($BATCH_ID)
    echo "[$(date +%H:%M:%S)] ✓ Submitted batch $BATCH_ID ($START-$END)"
done

echo ""
echo "=== All batches submitted. Monitoring completion and downloading ==="
echo ""

# Monitor and download as batches complete
ACTIVE_DOWNLOADS=0

while true; do
    # Check if all batches are downloaded
    ALL_DONE=true
    for BATCH_ID in "${ALL_BATCHES[@]}"; do
        if [ "${BATCH_STATUS[$BATCH_ID]}" != "downloaded" ] && [ "${BATCH_STATUS[$BATCH_ID]}" != "failed" ]; then
            ALL_DONE=false
            break
        fi
    done
    
    if [ "$ALL_DONE" = true ]; then
        break
    fi
    
    # Check for completed batches and start downloads
    for BATCH_ID in "${ALL_BATCHES[@]}"; do
        if [ "${BATCH_STATUS[$BATCH_ID]}" = "submitted" ]; then
            # Check if archive exists (job completed)
            if ssh bravhee@mentat001 "test -f ~/scenic_PhD/PreProcessing/test_batch_${BATCH_ID}.tar.gz" 2>/dev/null; then
                BATCH_STATUS[$BATCH_ID]="completed"
                
                # Start download in background if not at limit
                if [ $ACTIVE_DOWNLOADS -lt $PARALLEL_DOWNLOADS ]; then
                    download_batch $BATCH_ID &
                    ACTIVE_DOWNLOADS=$((ACTIVE_DOWNLOADS + 1))
                fi
            fi
        fi
        
        # Recount active downloads
        ACTIVE_DOWNLOADS=$(jobs -r | wc -l)
    done
    
    # Progress update
    DOWNLOADED=$(printf '%s\n' "${BATCH_STATUS[@]}" | grep -c "downloaded")
    TOTAL=${#ALL_BATCHES[@]}
    RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l" 2>/dev/null || echo "0")
    
    echo "[$(date +%H:%M:%S)] Progress: $DOWNLOADED/$TOTAL downloaded, $RUNNING jobs running, $ACTIVE_DOWNLOADS downloads active"
    
    sleep 15
done

# Wait for remaining downloads
wait

echo ""
echo "=== TEST Pipeline Complete (OPTIMIZED)! ==="
DOWNLOADED=$(printf '%s\n' "${BATCH_STATUS[@]}" | grep -c "downloaded")
FAILED=$(printf '%s\n' "${BATCH_STATUS[@]}" | grep -c "failed")
echo "Downloaded: $DOWNLOADED batches"
echo "Failed: $FAILED batches"
echo "Location: $LAPTOP_DIR"
