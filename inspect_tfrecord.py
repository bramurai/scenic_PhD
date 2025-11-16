#!/usr/bin/env python3
"""Inspect TFRecord files to see what features they contain."""

import tensorflow as tf
import sys

def inspect_tfrecord(file_path):
    """Print the features in a TFRecord file."""
    print(f"\nInspecting: {file_path}")
    print("=" * 80)
    
    try:
        # Read one example
        dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in dataset.take(1):
            example = tf.train.SequenceExample()
            example.ParseFromString(raw_record.numpy())
            
            print("\nContext Features:")
            for key in example.context.feature:
                print(f"  - {key}")
            
            print("\nFeature Lists:")
            for key in example.feature_lists.feature_list:
                print(f"  - {key}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_tfrecord(sys.argv[1])
    else:
        # Default: check first train file
        import glob
        train_files = sorted(glob.glob('/home/labuta/Documents/Bram/scenic_PhD/train_tfrecords_local/tar*_batch*/data-*.tfrecord'))
        if train_files:
            inspect_tfrecord(train_files[0])
        else:
            print("No TFRecord files found!")
