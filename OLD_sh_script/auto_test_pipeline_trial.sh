#!/bin/bash
# TEST VERSION: Automated workflow for TEST dataset - processes only first 500 videos

TOTAL_VIDEOS=500  # TEST ONLY - process first 500 videos
BATCH_SIZE=100  # Process 100 videos at a time (~550MB each with new resizing)
PARALLEL_JOBS=10  # Run 10 jobs simultaneously
LAPTOP_DIR="$1"  # Where to download on laptop

if [ -z "$LAPTOP_DIR" ]; then
    echo "Usage: bash auto_test_pipeline_trial.sh <laptop_download_dir>"
    echo "Example: bash auto_test_pipeline_trial.sh ~/Downloads/vggsound_test_trial"
    exit 1
fi

mkdir -p "$LAPTOP_DIR"

echo "=== VGGSound TEST Micro-Batch Pipeline (TRIAL) ==="
echo "Total videos: $TOTAL_VIDEOS (limited for testing)"
echo "Batch size: $BATCH_SIZE videos (~1-1.5GB per batch)"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Total batches: $((TOTAL_VIDEOS / BATCH_SIZE))"
echo ""

# Arrays to track jobs and batches
declare -a JOB_IDS
declare -a BATCH_IDS

# Submit all batches with parallel limit
for START in $(seq 0 $BATCH_SIZE $((TOTAL_VIDEOS - BATCH_SIZE))); do
    END=$((START + BATCH_SIZE))
    BATCH_ID=$(printf "%05d" $START)
    
    # Wait if we have too many jobs running
    RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l")
    while [ $RUNNING -ge $PARALLEL_JOBS ]; do
        echo "Waiting... ($RUNNING jobs running)"
        sleep 15
        RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l")
    done
    
    echo "Submitting batch $START-$END..."
    ssh bravhee@mentat001 "cd scenic_PhD && sbatch slurm_test_micro.sh $START $END"
    sleep 2
    
    JOB_ID=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h -o '%i' | tail -1")
    JOB_IDS+=($JOB_ID)
    BATCH_IDS+=($BATCH_ID)
    
    echo "✓ Submitted job $JOB_ID for batch $BATCH_ID"
done

echo ""
echo "All batches submitted. Waiting for completion..."
echo ""

# Wait for all jobs to complete
while ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h" | grep -q .; do
    RUNNING=$(ssh bravhee@mentat001 "squeue -u bravhee -n vggsound_test_micro -h | wc -l")
    echo "Still running: $RUNNING jobs..."
    sleep 30
done

echo ""
echo "All jobs completed! Downloading all batches..."
echo ""

# Download all batches
for BATCH_ID in "${BATCH_IDS[@]}"; do
    ARCHIVE="test_batch_${BATCH_ID}.tar.gz"
    echo "Downloading $ARCHIVE..."
    
    scp bravhee@mentat001:~/scenic_PhD/PreProcessing/$ARCHIVE "$LAPTOP_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "✓ Downloaded: $ARCHIVE"
        
        # Delete from cluster to free space
        ssh bravhee@mentat001 "rm ~/scenic_PhD/PreProcessing/$ARCHIVE; rm -rf ~/scenic_PhD/PreProcessing/tfrecords_test_micro/batch_${BATCH_ID}"
        echo "✓ Cleaned up cluster"
    else
        echo "✗ Download failed for $ARCHIVE"
    fi
    
    echo ""
done

echo ""
echo "=== TEST Pipeline TRIAL Complete! ==="
echo "All batches downloaded to: $LAPTOP_DIR"
echo "This was a trial run. To process full dataset, use auto_test_pipeline.sh"
echo "Extract all with: for f in $LAPTOP_DIR/*.tar.gz; do tar -xzf \$f -C /final/destination/test/; done"
