"""Compute spectrogram mean/stddev statistics from TFRecord dataset."""
import tensorflow as tf
import numpy as np
import glob
import os

# Path to dataset
DATASET_DIR = '/project/3026018.01/Models/MBT/Datasets/audioset_eval'

# Find all TFRecord files
tfrecord_pattern = os.path.join(DATASET_DIR, '**', '*.tfrecord')
tfrecord_files = glob.glob(tfrecord_pattern, recursive=True)
tfrecord_files.sort()

print(f"Found {len(tfrecord_files)} TFRecord files in {DATASET_DIR}")

# Count total samples and accumulate statistics
num_samples = 0
spec_sum = 0.0
spec_sum_sq = 0.0
spec_count = 0

print("\nComputing statistics over all samples...")

for tfr_idx, tfr_path in enumerate(tfrecord_files):
    if (tfr_idx + 1) % 5 == 0:
        print(f"  Processing file {tfr_idx + 1}/{len(tfrecord_files)}...")
    
    # Read all records from this file
    for raw_record in tf.data.TFRecordDataset([tfr_path]):
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        # Read spectrogram from feature_lists
        if 'spectrogram' in example.feature_lists.feature_list:
            spec_feature = example.feature_lists.feature_list['spectrogram'].feature
            
            # Each frame is stored separately
            for frame in spec_feature:
                spec_data = np.frombuffer(frame.bytes_list.value[0], dtype=np.uint8)
                # Decode from uint8 to float32 (stored as raw bytes)
                # The shape should be (100, 128, 3) for mel spectrograms
                # But we need to check actual encoding
                
                # Try different interpretations
                if len(spec_data) >= 100 * 128 * 3:
                    # Assume stored as float32 bytes
                    spec_float = np.frombuffer(frame.bytes_list.value[0], dtype=np.float32)
                    if len(spec_float) >= 100 * 128 * 3:
                        spec_frame = spec_float[:100*128*3].reshape(100, 128, 3)
                    else:
                        # Try reshaping available data
                        continue
                else:
                    continue
                
                # Accumulate statistics (only first channel if multi-channel)
                spec_values = spec_frame[:, :, 0].flatten()
                spec_sum += spec_values.sum()
                spec_sum_sq += (spec_values ** 2).sum()
                spec_count += len(spec_values)
        
        num_samples += 1
        
        if num_samples % 10 == 0:
            print(f"  Processed {num_samples} samples...", end='\r')

# Compute mean and stddev
if spec_count > 0:
    spec_mean = spec_sum / spec_count
    spec_variance = (spec_sum_sq / spec_count) - (spec_mean ** 2)
    spec_stddev = np.sqrt(spec_variance)
else:
    spec_mean = 0.0
    spec_stddev = 1.0

print(f"\n\n{'='*70}")
print(f"DATASET STATISTICS:")
print(f"{'='*70}")
print(f"Total samples: {num_samples}")
print(f"Total spectrogram values processed: {spec_count:,}")
print(f"\nLog-Mel Spectrogram Statistics:")
print(f"  Mean: {spec_mean:.6f}")
print(f"  Stddev: {spec_stddev:.6f}")
print(f"{'='*70}")

# Save to file
stats_file = os.path.join(DATASET_DIR, 'spec_statistics.txt')
with open(stats_file, 'w') as f:
    f.write(f"Dataset: {DATASET_DIR}\n")
    f.write(f"Total samples: {num_samples}\n")
    f.write(f"Spectrogram mean: {spec_mean:.6f}\n")
    f.write(f"Spectrogram stddev: {spec_stddev:.6f}\n")

print(f"\nStatistics saved to: {stats_file}")
print(f"\nAdd these to grid search:")
print(f"  'spec_mean': [0.0, 1.102, {spec_mean:.6f}]")
print(f"  'spec_stddev': [1.0, 2.762, {spec_stddev:.6f}]")
