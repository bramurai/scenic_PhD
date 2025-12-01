"""Analyze AudioSet TFRecords to count examples and classes.

This script reads TFRecords with multi-hot labels and reports:
- Total number of examples
- Number of active classes per example
- Most common classes
- Class distribution
"""

import os
import glob
from collections import Counter
import numpy as np
import tensorflow as tf
import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_dir', 'Datasets/audioset_eval', 'Directory containing TFRecord files.')
flags.DEFINE_string('audioset_labels_csv', 'Video_csvs/audioset_labels.csv', 'Path to audioset_labels.csv for class names.')
flags.DEFINE_integer('max_examples', None, 'Maximum number of examples to analyze (None = all).')


def parse_sequence_example(serialized_example):
    """Parse a SequenceExample from TFRecord."""
    context_features = {
        'clip/media_id': tf.io.FixedLenFeature([], tf.string),
        'clip/label/multi_hot': tf.io.VarLenFeature(tf.float32),
        'clip/label/num_active': tf.io.FixedLenFeature([], tf.int64),
    }
    
    sequence_features = {}
    
    context, sequence = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    
    return context


def main(argv):
    del argv
    
    # Load AudioSet labels
    print(f"Loading AudioSet labels from {FLAGS.audioset_labels_csv}")
    labels_df = pd.read_csv(FLAGS.audioset_labels_csv)
    index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
    index_to_mid = dict(zip(labels_df['index'], labels_df['mid']))
    num_classes = len(labels_df)
    print(f"Loaded {num_classes} AudioSet class labels")
    
    # Find all TFRecord files
    tfrecord_pattern = os.path.join(FLAGS.tfrecord_dir, '*.tfrecord')
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    print(f"\nFound {len(tfrecord_files)} TFRecord shards")
    
    if not tfrecord_files:
        print(f"No TFRecord files found in {FLAGS.tfrecord_dir}")
        return
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    # Analyze examples
    total_examples = 0
    class_counts = Counter()
    labels_per_example = []
    example_ids = []
    
    print("\nAnalyzing TFRecords...")
    for raw_record in dataset:
        if FLAGS.max_examples and total_examples >= FLAGS.max_examples:
            break
            
        try:
            context = parse_sequence_example(raw_record)
            
            # Get multi-hot label
            multi_hot = tf.sparse.to_dense(context['clip/label/multi_hot']).numpy()
            num_active = context['clip/label/num_active'].numpy()
            media_id = context['clip/media_id'].numpy().decode('utf-8')
            
            # Count active classes
            active_indices = np.where(multi_hot > 0)[0]
            labels_per_example.append(len(active_indices))
            example_ids.append(media_id)
            
            # Update class counts
            for idx in active_indices:
                class_counts[int(idx)] += 1
            
            total_examples += 1
            
            if total_examples % 1000 == 0:
                print(f"Processed {total_examples} examples...")
                
        except Exception as e:
            print(f"Error parsing example {total_examples}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total examples: {total_examples}")
    print(f"Average labels per example: {np.mean(labels_per_example):.2f}")
    print(f"Min labels per example: {np.min(labels_per_example)}")
    print(f"Max labels per example: {np.max(labels_per_example)}")
    print(f"Unique classes present: {len(class_counts)}")
    
    # Show most common classes
    print(f"\n{'='*60}")
    print(f"TOP 20 MOST COMMON CLASSES")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Count':<8} {'Index':<8} {'MID':<20} {'Display Name'}")
    print(f"{'-'*60}")
    
    for rank, (class_idx, count) in enumerate(class_counts.most_common(20), 1):
        name = index_to_name.get(class_idx, 'Unknown')
        mid = index_to_mid.get(class_idx, 'Unknown')
        print(f"{rank:<6} {count:<8} {class_idx:<8} {mid:<20} {name}")
    
    # Show least common classes
    print(f"\n{'='*60}")
    print(f"LEAST COMMON CLASSES (showing classes with < 10 examples)")
    print(f"{'='*60}")
    
    rare_classes = [(idx, count) for idx, count in class_counts.items() if count < 10]
    rare_classes.sort(key=lambda x: x[1])
    
    if rare_classes:
        print(f"{'Count':<8} {'Index':<8} {'MID':<20} {'Display Name'}")
        print(f"{'-'*60}")
        for class_idx, count in rare_classes[:20]:
            name = index_to_name.get(class_idx, 'Unknown')
            mid = index_to_mid.get(class_idx, 'Unknown')
            print(f"{count:<8} {class_idx:<8} {mid:<20} {name}")
    else:
        print("No rare classes found (all classes have >= 10 examples)")
    
    # Show example distribution
    print(f"\n{'='*60}")
    print(f"LABELS PER EXAMPLE DISTRIBUTION")
    print(f"{'='*60}")
    
    label_dist = Counter(labels_per_example)
    for num_labels in sorted(label_dist.keys())[:10]:
        count = label_dist[num_labels]
        percentage = 100 * count / total_examples
        print(f"{num_labels} labels: {count} examples ({percentage:.1f}%)")
    
    # Show a few example IDs
    print(f"\n{'='*60}")
    print(f"SAMPLE EXAMPLE IDs (first 10)")
    print(f"{'='*60}")
    for i, ex_id in enumerate(example_ids[:10], 1):
        print(f"{i}. {ex_id}")
    
    # Save detailed class statistics
    output_file = os.path.join(FLAGS.tfrecord_dir, 'class_statistics.csv')
    class_stats = []
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        name = index_to_name.get(class_idx, 'Unknown')
        mid = index_to_mid.get(class_idx, 'Unknown')
        class_stats.append({
            'index': class_idx,
            'mid': mid,
            'display_name': name,
            'count': count,
            'percentage': 100 * count / total_examples
        })
    
    stats_df = pd.DataFrame(class_stats)
    stats_df.to_csv(output_file, index=False)
    print(f"\nDetailed class statistics saved to: {output_file}")


if __name__ == '__main__':
    app.run(main)
