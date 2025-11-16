#!/usr/bin/env python3
"""Inspect VGGSound TFRecord structure to verify compatibility with config."""

import tensorflow as tf
import numpy as np
import glob
import os

def inspect_tfrecord(tfrecord_path, num_samples=3):
    """Inspect TFRecord file structure."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {tfrecord_path}")
    print(f"{'='*80}\n")
    
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    for idx, raw_record in enumerate(dataset.take(num_samples)):
        print(f"\n--- Sample {idx + 1} ---")
        
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        # Check context features (metadata)
        print("\n📋 Context Features:")
        for key in example.context.feature:
            feature = example.context.feature[key]
            if feature.HasField('bytes_list'):
                val = feature.bytes_list.value[0] if len(feature.bytes_list.value) > 0 else b''
                print(f"  {key}: bytes (length={len(val)})")
                # Try to decode label
                if 'label' in key.lower() or 'class' in key.lower():
                    try:
                        decoded = val.decode('utf-8')
                        print(f"    → Decoded: '{decoded}'")
                    except:
                        print(f"    → Raw bytes: {val[:50]}...")
            elif feature.HasField('int64_list'):
                vals = list(feature.int64_list.value)
                print(f"  {key}: int64 = {vals}")
            elif feature.HasField('float_list'):
                vals = list(feature.float_list.value)
                print(f"  {key}: float = {vals[:5]}{'...' if len(vals) > 5 else ''}")
        
        # Check feature lists (sequences)
        print("\n🎬 Feature Lists (Sequences):")
        for key in example.feature_lists.feature_list:
            feature_list = example.feature_lists.feature_list[key]
            num_frames = len(feature_list.feature)
            
            if num_frames > 0:
                first_feature = feature_list.feature[0]
                
                if first_feature.HasField('bytes_list'):
                    # Decode image/video frame
                    if 'image' in key.lower() or 'rgb' in key.lower():
                        try:
                            frame_bytes = first_feature.bytes_list.value[0]
                            # Try to decode as JPEG
                            frame = tf.image.decode_jpeg(frame_bytes)
                            shape = frame.shape
                            print(f"  {key}: {num_frames} frames, each {shape} (decoded JPEG)")
                        except:
                            print(f"  {key}: {num_frames} frames (bytes, couldn't decode)")
                    else:
                        val_len = len(first_feature.bytes_list.value[0]) if len(first_feature.bytes_list.value) > 0 else 0
                        print(f"  {key}: {num_frames} frames (bytes, {val_len} bytes each)")
                
                elif first_feature.HasField('float_list'):
                    num_values = len(first_feature.float_list.value)
                    values = list(first_feature.float_list.value)
                    
                    # This is likely the spectrogram!
                    print(f"  {key}: {num_frames} frames, {num_values} values per frame")
                    
                    # Try to infer shape
                    if num_values == 128:
                        print(f"    → Likely shape per frame: (1, 128) - MATCHES VGGSound config!")
                        print(f"    → Total spectrogram: ({num_frames}, 128)")
                    elif num_values == 12800:  # 100 * 128
                        print(f"    → Likely shape per frame: (100, 128) - MATCHES AudioSet config!")
                        print(f"    → Total spectrogram: ({num_frames * 100}, 128)")
                    else:
                        print(f"    → Unknown structure, may need config adjustment")
                    
                    # Show first few values
                    print(f"    → Sample values: {values[:5]}")
                
                elif first_feature.HasField('int64_list'):
                    num_values = len(first_feature.int64_list.value)
                    values = list(first_feature.int64_list.value)
                    print(f"  {key}: {num_frames} frames, {num_values} values per frame")
                    print(f"    → Sample values: {values[:5]}")
        
        print(f"\n{'-'*80}")

def check_label_diversity(tfrecord_paths, num_samples=100):
    """Check if all samples have the same label."""
    print(f"\n{'='*80}")
    print("CHECKING LABEL DIVERSITY")
    print(f"{'='*80}\n")
    
    labels_found = set()
    label_counts = {}
    
    # Sample from multiple files
    dataset = tf.data.TFRecordDataset(tfrecord_paths[:5])  # Check first 5 files
    
    for idx, raw_record in enumerate(dataset.take(num_samples)):
        if idx >= num_samples:
            break
            
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        # Look for label in context features
        for key in example.context.feature:
            if 'label' in key.lower() or 'class' in key.lower():
                feature = example.context.feature[key]
                
                if feature.HasField('bytes_list'):
                    label = feature.bytes_list.value[0].decode('utf-8') if len(feature.bytes_list.value) > 0 else ''
                    labels_found.add(label)
                    label_counts[label] = label_counts.get(label, 0) + 1
                elif feature.HasField('int64_list'):
                    label = tuple(feature.int64_list.value)
                    labels_found.add(label)
                    label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Samples checked: {min(idx + 1, num_samples)}")
    print(f"Unique labels found: {len(labels_found)}")
    
    if len(labels_found) == 1:
        print(f"\n⚠️  WARNING: All samples have the SAME label!")
        print(f"   Label: {list(labels_found)[0]}")
        print(f"\n   This means your TFRecords were NOT created correctly.")
        print(f"   Each sample should have its own class label.")
        print(f"\n   You MUST remake the TFRecords with proper labels.")
    else:
        print(f"\n✓ Labels look diverse!")
        print(f"\nLabel distribution (top 10):")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels[:10]:
            print(f"  {label}: {count} samples")
    
    return len(labels_found) > 1

def main():
    """Main inspection routine."""
    print("\n" + "="*80)
    print("VGGSOUND TFRECORD INSPECTOR")
    print("="*80)
    
    # Find TFRecord files
    base_dir = '/home/labuta/Documents/Bram/scenic_PhD'
    train_pattern = os.path.join(base_dir, 'train_tfrecords_local/tar*_batch*/data-*-of-*.tfrecord')
    
    train_files = sorted(glob.glob(train_pattern))
    
    if not train_files:
        print(f"\n❌ No TFRecord files found at: {train_pattern}")
        print("\nPlease update the path in this script.")
        return
    
    print(f"\nFound {len(train_files)} TFRecord files")
    print(f"First file: {train_files[0]}")
    
    # Inspect structure of first file
    print("\n" + "="*80)
    print("PART 1: STRUCTURE INSPECTION")
    print("="*80)
    inspect_tfrecord(train_files[0], num_samples=2)
    
    # Check label diversity
    print("\n" + "="*80)
    print("PART 2: LABEL DIVERSITY CHECK")
    print("="*80)
    labels_ok = check_label_diversity(train_files, num_samples=100)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. SPECTROGRAM STRUCTURE:")
    print("   - Check the output above for 'WAVEFORM/feature/floats'")
    print("   - If it shows '(1, 128)' per frame → Keep num_spec_frames=800, spec_shape=(1, 128)")
    print("   - If it shows '(100, 128)' per frame → Change to num_spec_frames=8, spec_shape=(100, 128)")
    
    print("\n2. LABEL ISSUE:")
    if not labels_ok:
        print("   ❌ CRITICAL: You MUST remake TFRecords with proper per-sample labels!")
        print("   - Current TFRecords have all samples with the same label")
        print("   - Training on these will not work properly")
        print("   - Check your TFRecord creation script - labels are not being written correctly")
    else:
        print("   ✓ Labels look good - diverse across samples")
    
    print("\n3. NEXT STEPS:")
    if not labels_ok:
        print("   a) Fix the TFRecord creation script to include correct labels")
        print("   b) Regenerate all TFRecords")
        print("   c) Re-run this inspection script to verify")
    else:
        print("   a) Adjust config based on spectrogram structure above")
        print("   b) Start training!")

if __name__ == '__main__':
    main()
