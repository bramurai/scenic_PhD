#!/bin/bash
#SBATCH --job-name=regen_tfrecords
#SBATCH --output=regen_tfrecords_%j.out
#SBATCH --error=regen_tfrecords_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G


# Regenerate TFRecords from downloaded videos in temp_downloads/
# This script uses the existing downloaded segments and extracts from 0-10s instead of CSV timestamps

set -e

echo "========================================="
echo "Regenerating TFRecords from temp_downloads/"
echo "========================================="
echo "Start time: $(date)"
echo ""

# Activate environment
source activate scenic_preprocessing

# Set working directory
cd /project/3026018.01/Models/MBT

# Enable multi-threading for FFmpeg and numpy
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# Delete old TFRecords to regenerate from scratch
echo "Deleting old TFRecords..."
rm -rf Datasets/audioset_eval/data-*.tfrecord
echo "Old TFRecords deleted."
echo ""

# Regenerate eval set (using existing videos in temp_downloads/)
echo "Regenerating AudioSet eval TFRecords..."
echo "Logs will update every 10 videos processed..."
echo ""

# Run with unbuffered output for real-time logging
python3 -u download_and_preprocess.py \
    --csv_path=Video_csvs/audioset_eval.csv \
    --output_path=Datasets/audioset_eval \
    --num_shards=10 \
    --audioset_labels_csv=Video_csvs/audioset_labels.csv \
    --local_videos_dir=downloaded_videos \
    --require_local=False \
    --local_are_clips=True \
    --download_full_segment=True \
    --save_progress_every=10 \
    --target_fps=25 \
    --audio_sample_rate=16000 \
    --n_mels=128 

echo ""
echo "========================================="
echo "TFRecord regeneration complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Verifying first TFRecord..."
python3 <<'EOF'
import tensorflow as tf
import numpy as np
import glob

tfrecords = sorted(glob.glob('Datasets/audioset_eval/**/*.tfrecord', recursive=True))
if tfrecords:
    print(f"\nFirst TFRecord: {tfrecords[0]}")
    
    for raw_record in tf.data.TFRecordDataset([tfrecords[0]]).take(1):
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        # Get label
        multi_hot = example.context.feature['clip/label/multi_hot'].float_list.value
        active = np.where(np.array(multi_hot) > 0)[0]
        print(f"Active label indices: {active}")
        
        # Check RGB frames
        if 'image/encoded' in example.feature_lists.feature_list:
            num_frames = len(example.feature_lists.feature_list['image/encoded'].feature)
            print(f"RGB frames: {num_frames}")
        
        # Check audio
        if 'WAVEFORM/feature/floats' in example.feature_lists.feature_list:
            num_audio = len(example.feature_lists.feature_list['WAVEFORM/feature/floats'].feature)
            print(f"Audio frames: {num_audio}")
else:
    print("ERROR: No TFRecords found!")
EOF

echo ""
echo "You can now re-run the extraction script:"
echo "  sbatch run_mbt_extraction.sh"
