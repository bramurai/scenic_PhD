#!/usr/bin/env python3
"""Inspect TFRecord files to see the actual shape of features."""

import tensorflow as tf
import numpy as np

def inspect_waveform_shape(file_path):
    """Print the shape of WAVEFORM features in a TFRecord file."""
    print(f"\nInspecting: {file_path}")
    print("=" * 80)
    
    try:
        dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in dataset.take(1):
            example = tf.train.SequenceExample()
            example.ParseFromString(raw_record.numpy())
            
            # Get WAVEFORM feature
            waveform_features = example.feature_lists.feature_list['WAVEFORM/feature/floats']
            
            print(f"\nNumber of frames: {len(waveform_features.feature)}")
            
            if len(waveform_features.feature) > 0:
                first_frame = waveform_features.feature[0]
                print(f"Values per frame: {len(first_frame.float_list.value)}")
                
                # Calculate total values
                total_values = sum(len(f.float_list.value) for f in waveform_features.feature)
                print(f"Total values: {total_values}")
                
                # Infer shape
                num_frames = len(waveform_features.feature)
                values_per_frame = len(waveform_features.feature[0].float_list.value)
                print(f"\nInferred shape: [{num_frames}, {values_per_frame}]")
                
                # Try to figure out mel bins
                if values_per_frame % 100 == 0:
                    mel_bins = values_per_frame // 100
                    print(f"Possible interpretation: [100, {mel_bins}] per frame, {num_frames} frames")
                if values_per_frame % 128 == 0:
                    mel_bins = values_per_frame // 128
                    print(f"Possible interpretation: [{mel_bins}, 128] per frame, {num_frames} frames")
                
                # Check if this could be time x mel_bins
                print(f"\nIf stored as [time, mel_bins]:")
                print(f"  Total frames in TFRecord: {num_frames}")
                print(f"  Values per frame: {values_per_frame}")
                print(f"  This means each frame has shape: ({values_per_frame},)")
                print(f"  Which should be reshaped to: ({values_per_frame}, 1) or kept as ({values_per_frame},)")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import glob
    train_files = sorted(glob.glob('/home/labuta/Documents/Bram/scenic_PhD/train_tfrecords_local/tar*_batch*/data-*.tfrecord'))
    if train_files:
        inspect_waveform_shape(train_files[0])
    else:
        print("No TFRecord files found!")
