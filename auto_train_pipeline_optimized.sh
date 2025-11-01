#!/bin/bash
# OPTIMIZED: Download batches immediately as they complete (parallel download) - TRAINING

TOTAL_VIDEOS=500 # 183971  # Training set size
BATCH_SIZE=50  # Process 200 videos at a time (~1.1GB each)
PARALLEL_JOBS=25  # Run 25 jobs simultaneously
PARALLEL_DOWNLOADS=25  # Download 25 batches simultaneously (1 Gbps connection can handle it!)
LAPTOP_DIR="$1"

if [ -z "$LAPTOP_DIR" ]; then
    echo "Usage: bash auto_train_pipeline_optimized.sh <laptop_download_dir>"
    exit 1
fi

mkdir -p "$LAPTOP_DIR"

echo "=== VGGSound TRAINING Pipeline (OPTIMIZED) ==="
echo "Total videos: $TOTAL_VIDEOS"
echo "Batch size: $BATCH_SIZE videos"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Parallel downloads: $PARALLEL_DOWNLOADS"
echo "Total batches: $((TOTAL_VIDEOS / BATCH_SIZE))"
echo ""

# Track submitted batches
declare -A BATCH_STATUS
declare -a ALL_BATCHES

# Function to download a batch in background
download_batch() {
    local BATCH_ID=$1
    local ARCHIVE="train_batch_${BATCH_ID}.tar.gz"
    
    echo "[$(date +%H:%M:%S)] Downloading $ARCHIVE..."
    
    if scp -q bravhee@mentat001.dccn.nl:~/scenic_PhD/PreProcessing/$ARCHIVE "$LAPTOP_DIR/" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ✓ Downloaded: $ARCHIVE"
        
        # Delete from cluster
        ssh bravhee@mentat001.dccn.nl "rm ~/scenic_PhD/PreProcessing/$ARCHIVE; rm -rf ~/scenic_PhD/PreProcessing/tfrecords_train_micro/batch_${BATCH_ID}" 2>/dev/null
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
    RUNNING=$(ssh bravhee@mentat001.dccn.nl "squeue -u bravhee -n vggsound_train_micro -h | wc -l")
    while [ $RUNNING -ge $PARALLEL_JOBS ]; do
        sleep 10
        RUNNING=$(ssh bravhee@mentat001.dccn.nl "squeue -u bravhee -n vggsound_train_micro -h | wc -l")
    done
    
    ssh bravhee@mentat001.dccn.nl "cd scenic_PhD && sbatch slurm_train_micro.sh $START $END" > /dev/null 2>&1
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
        STATUS="${BATCH_STATUS[$BATCH_ID]}"
        if [ "$STATUS" != "downloaded" ] && [ "$STATUS" != "failed" ]; then
            ALL_DONE=false
            break
        fi
    done
    
    if [ "$ALL_DONE" = true ]; then
        break
    fi
    
    # Check for completed batches and start downloads
    for BATCH_ID in "${ALL_BATCHES[@]}"; do
        if [ "${BATCH_STATUS[$BATCH_ID]}" = "submitted" ] || [ "${BATCH_STATUS[$BATCH_ID]}" = "completed" ]; then
            # Check if archive exists (job completed)
            if ssh bravhee@mentat001.dccn.nl "test -f ~/scenic_PhD/PreProcessing/train_batch_${BATCH_ID}.tar.gz" 2>/dev/null; then
                # Only mark as completed if not already downloading
                if [ "${BATCH_STATUS[$BATCH_ID]}" = "submitted" ]; then
                    BATCH_STATUS[$BATCH_ID]="completed"
                fi
                
                # Start download in background if not already downloading and not at limit
                if [ "${BATCH_STATUS[$BATCH_ID]}" = "completed" ] && [ $ACTIVE_DOWNLOADS -lt $PARALLEL_DOWNLOADS ]; then
                    BATCH_STATUS[$BATCH_ID]="downloading"
                    download_batch $BATCH_ID &
                    ACTIVE_DOWNLOADS=$((ACTIVE_DOWNLOADS + 1))
                fi
            fi
        fi
        
        # Recount active downloads
        ACTIVE_DOWNLOADS=$(jobs -r | wc -l)
    done
    
    # Check for failed batches (jobs completed but no archive) - only after all jobs are done
    RUNNING=$(ssh bravhee@mentat001.dccn.nl "squeue -u bravhee -n vggsound_train_micro -h | wc -l" 2>/dev/null || echo "0")
    if [ $RUNNING -eq 0 ]; then
        for BATCH_ID in "${ALL_BATCHES[@]}"; do
            if [ "${BATCH_STATUS[$BATCH_ID]}" = "submitted" ]; then
                # No archive and no jobs running = failed
                if ! ssh bravhee@mentat001.dccn.nl "test -f ~/scenic_PhD/PreProcessing/train_batch_${BATCH_ID}.tar.gz" 2>/dev/null; then
                    echo "[$(date +%H:%M:%S)] ✗ Batch $BATCH_ID failed (no archive created)"
                    BATCH_STATUS[$BATCH_ID]="failed"
                fi
            fi
        done
    fi
    
    # Progress update - count actual downloaded files since background jobs can't update the array
    DOWNLOADED=0
    for BATCH_ID in "${ALL_BATCHES[@]}"; do
        if [ -f "$LAPTOP_DIR/train_batch_${BATCH_ID}.tar.gz" ]; then
            BATCH_STATUS[$BATCH_ID]="downloaded"
            DOWNLOADED=$((DOWNLOADED + 1))
        fi
    done
    
    TOTAL=${#ALL_BATCHES[@]}
    RUNNING=$(ssh bravhee@mentat001.dccn.nl "squeue -u bravhee -n vggsound_train_micro -h | wc -l" 2>/dev/null || echo "0")
    
    echo "[$(date +%H:%M:%S)] Progress: $DOWNLOADED/$TOTAL downloaded, $RUNNING jobs running, $ACTIVE_DOWNLOADS downloads active"
    
    sleep 15
done

# Wait for remaining downloads
wait

echo ""
echo "=== TRAINING Pipeline Complete (OPTIMIZED)! ==="

# Count actual downloaded files
DOWNLOADED=0
FAILED=0
for BATCH_ID in "${ALL_BATCHES[@]}"; do
    if [ -f "$LAPTOP_DIR/train_batch_${BATCH_ID}.tar.gz" ]; then
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo "Downloaded: $DOWNLOADED batches"
echo "Failed: $FAILED batches"
echo "Location: $LAPTOP_DIR"
