#!/usr/bin/env python3
"""
Fix label indices in existing TFRecords.
Converts from old custom indices (0-76) to correct AudioSet indices (0-526).
"""

import tensorflow as tf
import os
import glob
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'Datasets/audioset_eval_100', 
                    'Directory containing TFRecords to fix')
flags.DEFINE_string('output_dir', None,
                    'Output directory for fixed TFRecords. If None, overwrites input files.')
flags.DEFINE_string('old_mapping', 'Datasets/audioset_eval_100/label_mapping_OLD_WRONG.txt',
                    'Path to old label mapping file')
flags.DEFINE_string('new_mapping', 'Datasets/audioset_eval_100/label_mapping.txt',
                    'Path to new corrected label mapping file')


def load_label_mapping(filepath):
    """Load label mapping from file.
    
    Returns:
        Dict mapping index -> label_name
    """
    mapping = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                idx = int(parts[0])
                name = '\t'.join(parts[1:])  # In case label has tabs
                mapping[idx] = name
    return mapping


def create_index_conversion_map(old_mapping, new_mapping):
    """Create mapping from old indices to new indices.
    
    Args:
        old_mapping: Dict {old_index: label_name}
        new_mapping: Dict {new_index: label_name}
    
    Returns:
        Dict {old_index: new_index}
    """
    # Invert new_mapping to get {label_name: new_index}
    name_to_new_idx = {name: idx for idx, name in new_mapping.items()}
    
    # Create conversion map
    conversion = {}
    for old_idx, label_name in old_mapping.items():
        if label_name in name_to_new_idx:
            new_idx = name_to_new_idx[label_name]
            conversion[old_idx] = new_idx
            logging.info(f'{label_name}: {old_idx} -> {new_idx}')
        else:
            logging.warning(f'Label "{label_name}" not found in new mapping!')
    
    return conversion


def fix_tfrecord(input_path, output_path, conversion_map, num_classes=527):
    """Fix labels in a single TFRecord file.
    
    Args:
        input_path: Path to input TFRecord
        output_path: Path to output TFRecord
        conversion_map: Dict mapping old indices to new indices
        num_classes: Number of classes in new one-hot encoding
    """
    import numpy as np
    
    logging.info(f'Processing {os.path.basename(input_path)}...')
    
    # Read and write TFRecords
    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for record in tf.data.TFRecordDataset(input_path):
            # Parse the record
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            
            # Get the current label
            label_feature = example.features.feature['label']
            
            if label_feature.int64_list.value:
                # Label is stored as int64
                old_label_idx = int(label_feature.int64_list.value[0])
                
                if old_label_idx in conversion_map:
                    new_label_idx = conversion_map[old_label_idx]
                    
                    # Update the label
                    example.features.feature['label'].int64_list.value[0] = new_label_idx
                    
                    if count == 0:
                        logging.info(f'  Example: old label {old_label_idx} -> new label {new_label_idx}')
                else:
                    logging.warning(f'  Old label {old_label_idx} not in conversion map!')
            
            elif label_feature.float_list.value:
                # Label is stored as one-hot float array
                old_one_hot = np.array(label_feature.float_list.value)
                old_label_idx = int(np.argmax(old_one_hot))
                
                if old_label_idx in conversion_map:
                    new_label_idx = conversion_map[old_label_idx]
                    
                    # Create new one-hot encoding
                    new_one_hot = np.zeros(num_classes, dtype=np.float32)
                    new_one_hot[new_label_idx] = 1.0
                    
                    # Update the feature
                    example.features.feature['label'].float_list.value[:] = new_one_hot.tolist()
                    
                    if count == 0:
                        logging.info(f'  Example: old label {old_label_idx} -> new label {new_label_idx}')
                else:
                    logging.warning(f'  Old label {old_label_idx} not in conversion map!')
            
            # Write the modified example
            writer.write(example.SerializeToString())
            count += 1
    
    logging.info(f'  Processed {count} examples')
    return count


def main(argv):
    del argv  # Unused
    
    logging.info('='*80)
    logging.info('TFRecord Label Fixer')
    logging.info('='*80)
    
    # Load label mappings
    logging.info('\n[1/4] Loading label mappings...')
    old_mapping = load_label_mapping(FLAGS.old_mapping)
    new_mapping = load_label_mapping(FLAGS.new_mapping)
    
    logging.info(f'  Old mapping: {len(old_mapping)} classes')
    logging.info(f'  New mapping: {len(new_mapping)} classes')
    
    # Create conversion map
    logging.info('\n[2/4] Creating conversion map...')
    conversion_map = create_index_conversion_map(old_mapping, new_mapping)
    logging.info(f'  Mapped {len(conversion_map)} labels')
    
    # Find TFRecord files
    logging.info('\n[3/4] Finding TFRecord files...')
    tfrecord_pattern = os.path.join(FLAGS.input_dir, '*.tfrecord*')
    tfrecord_files = sorted(glob.glob(tfrecord_pattern))
    
    if not tfrecord_files:
        logging.error(f'No TFRecord files found matching: {tfrecord_pattern}')
        return
    
    logging.info(f'  Found {len(tfrecord_files)} TFRecord files')
    
    # Determine output directory
    if FLAGS.output_dir is None:
        output_dir = FLAGS.input_dir
        logging.info('  Output: Overwriting input files')
    else:
        output_dir = FLAGS.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f'  Output: {output_dir}')
    
    # Process each file
    logging.info('\n[4/4] Processing TFRecords...')
    total_examples = 0
    
    for tfrecord_file in tfrecord_files:
        input_path = tfrecord_file
        
        # Create output path
        basename = os.path.basename(tfrecord_file)
        if FLAGS.output_dir is None:
            # Overwrite: use temporary file then rename
            output_path = tfrecord_file + '.tmp'
        else:
            output_path = os.path.join(output_dir, basename)
        
        # Fix the TFRecord
        count = fix_tfrecord(input_path, output_path, conversion_map)
        total_examples += count
        
        # If overwriting, replace original file
        if FLAGS.output_dir is None:
            os.replace(output_path, input_path)
            logging.info(f'  ✓ Updated {basename}')
        else:
            logging.info(f'  ✓ Created {basename}')
    
    logging.info('\n' + '='*80)
    logging.info('Complete!')
    logging.info(f'  Processed {len(tfrecord_files)} files')
    logging.info(f'  Total examples: {total_examples}')
    logging.info('='*80)
    
    # Show some example conversions
    logging.info('\nExample label conversions:')
    for old_idx, new_idx in list(conversion_map.items())[:10]:
        label_name = old_mapping[old_idx]
        logging.info(f'  "{label_name}": {old_idx} -> {new_idx}')


if __name__ == '__main__':
    app.run(main)
