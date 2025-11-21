#!/usr/bin/env python3
"""Diagnostic script to profile data loading performance."""

import time
import tensorflow as tf
from scenic.projects.mbt.configs.audioset import vggsound_base
from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset

def profile_data_loading():
    """Profile different stages of data loading."""
    
    # Get config
    config = vggsound_base.get_config()
    dataset_configs = config.dataset_configs
    
    print("="*80)
    print("DATA LOADING PERFORMANCE DIAGNOSIS")
    print("="*80)
    print(f"\nDataset: VGGSound")
    print(f"Batch size: {config.batch_size}")
    print(f"Num spec frames: {dataset_configs.num_spec_frames}")
    print(f"Num RGB frames: {dataset_configs.num_frames}")
    print(f"Prefetch to host: {dataset_configs.get('prefetch_to_host', 'NOT SET')}")
    print(f"Prefetch to device: {dataset_configs.get('prefetch_to_device', 'NOT SET')}")
    
    # Create dataset
    print("\n" + "="*80)
    print("CREATING DATASET...")
    print("="*80)
    
    t0 = time.time()
    dataset = audiovisual_tfrecord_dataset.get_dataset(
        batch_size=config.batch_size,
        eval_batch_size=config.batch_size,
        num_shards=1,
        dtype_str='float32',
        shuffle_seed=0,
        rng=None,
        dataset_configs=dataset_configs,
        dataset_service_address=None
    )
    t1 = time.time()
    print(f"Dataset creation took: {t1-t0:.2f}s")
    
    # Test iteration speed
    print("\n" + "="*80)
    print("PROFILING ITERATION SPEED...")
    print("="*80)
    
    iter_times = []
    print("\nTiming first 20 batches:")
    
    for i, batch in enumerate(dataset.train_iter):
        if i >= 20:
            break
        
        t0 = time.time()
        # Force numpy conversion to ensure data is actually loaded
        _ = {k: v.shape for k, v in batch['inputs'].items()}
        t1 = time.time()
        
        iter_time = t1 - t0
        iter_times.append(iter_time)
        
        status = "⚠️  SLOW" if iter_time > 0.5 else "✓ OK"
        print(f"  Batch {i+1:2d}: {iter_time:.4f}s {status}")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    import numpy as np
    iter_times = np.array(iter_times)
    
    print(f"\nIteration times (excluding first batch compilation):")
    print(f"  Mean:   {np.mean(iter_times[1:]):.4f}s")
    print(f"  Median: {np.median(iter_times[1:]):.4f}s")
    print(f"  Min:    {np.min(iter_times[1:]):.4f}s")
    print(f"  Max:    {np.max(iter_times[1:]):.4f}s")
    print(f"  Std:    {np.std(iter_times[1:]):.4f}s")
    
    avg_time = np.mean(iter_times[1:])
    steps_per_sec = 1.0 / avg_time
    
    print(f"\n  Steps/sec: {steps_per_sec:.2f}")
    print(f"  ETA for 1M steps: {1_000_000 / steps_per_sec / 86400:.1f} days")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if avg_time > 1.0:
        print("\n⚠️  Data loading is VERY SLOW (>1s per batch)")
        print("\nPossible issues:")
        print("  1. Large spectrogram size (800 frames) - consider reducing if possible")
        print("  2. SpecAugment on large spectrograms - very CPU intensive")
        print("  3. TFRecord decompression bottleneck")
        print("  4. Disk I/O bottleneck (check with `iostat -x 1`)")
        print("\nTry:")
        print("  - Reduce spec_augment_params.time_mask_count from 2 to 1")
        print("  - Reduce num_spec_frames if your task allows it")
        print("  - Move TFRecords to faster storage (SSD/NVMe)")
        print("  - Use tf.data.AUTOTUNE prefetch (already enabled)")
    elif avg_time > 0.5:
        print("\n⚠️  Data loading is moderate (0.5-1s per batch)")
        print("  Consider the optimizations above for faster training")
    else:
        print("\n✓ Data loading looks good!")
        print(f"  Average time: {avg_time:.4f}s")

if __name__ == '__main__':
    profile_data_loading()
