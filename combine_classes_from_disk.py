#!/usr/bin/env python3
"""Combine class-averaged activations from disk without loading all data.

This script shows how to:
1. Load only specific classes from disk
2. Combine classes (e.g., "Music" + "Speech" → "Audio Content")
3. Compute custom RDMs with different class groupings
4. Never load all data into RAM (stays < 5GB)

Usage examples:
  # Combine Music + Speech + Singing
  python combine_classes_from_disk.py \
    --averaged_dir=audioset_analysis_AV/averaged_activations \
    --class_indices=10,11,12 \
    --output_name=audio_content \
    --output_dir=combined_classes

  # Create a hierarchical grouping
  python combine_classes_from_disk.py \
    --config=groupings.json \
    --output_dir=hierarchical_classes
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_string('averaged_dir', None, 'Path to averaged_activations directory')
flags.DEFINE_string('audioset_labels_csv', 'Video_csvs/audioset_labels.csv', 'Path to AudioSet labels')
flags.DEFINE_string('config', None, 'Path to JSON config file with groupings (optional)')
flags.DEFINE_string('class_indices', None, 'Comma-separated list of class indices to combine')
flags.DEFINE_string('output_name', 'combined_class', 'Name for combined class')
flags.DEFINE_string('output_dir', 'combined_classes', 'Output directory')

flags.mark_flag_as_required('averaged_dir')


def load_activation_from_disk(class_idx, act_name, averaged_dir):
  """Load a single averaged activation from disk (one operation, low memory).
  
  Returns:
    activation: np.array or None if not found
  """
  path = os.path.join(averaged_dir, f'class_{class_idx}_{act_name}.npy')
  if os.path.exists(path):
    return np.load(path)
  return None


def get_activation_names(averaged_dir):
  """Get list of activation names from first file found."""
  for filename in os.listdir(averaged_dir):
    if filename.endswith('.npy'):
      # Extract activation name from "class_X_NAME.npy"
      parts = filename.replace('.npy', '').split('_', 2)
      if len(parts) >= 3:
        return [parts[2]]  # Will be completed by scanning all files
  
  # Scan all files to get all activation names
  act_names = set()
  for filename in os.listdir(averaged_dir):
    if filename.endswith('.npy'):
      parts = filename.replace('.npy', '').split('_', 2)
      if len(parts) >= 3:
        act_names.add(parts[2])
  
  return sorted(list(act_names))


def combine_classes(class_indices, averaged_dir, activation_names):
  """Combine multiple classes by averaging their activations.
  
  Args:
    class_indices: list of class indices to combine
    averaged_dir: path to averaged_activations directory
    activation_names: list of activation names to combine
    
  Returns:
    combined_activations: dict of combined activation arrays
  """
  combined = {}
  
  for act_name in activation_names:
    activations = []
    
    for class_idx in class_indices:
      act = load_activation_from_disk(class_idx, act_name, averaged_dir)
      if act is not None:
        activations.append(act)
    
    if activations:
      # Average the activations
      combined_act = np.mean(np.array(activations), axis=0)
      combined[act_name] = combined_act
  
  return combined


def save_combined_class(combined_activations, class_name, output_dir, metadata):
  """Save combined class as individual .npy files in output directory."""
  os.makedirs(output_dir, exist_ok=True)
  
  for act_name, act_array in combined_activations.items():
    path = os.path.join(output_dir, f'{class_name}_{act_name}.npy')
    np.save(path, act_array)
  
  # Save metadata
  metadata_path = os.path.join(output_dir, f'{class_name}_metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('Combining Classes from Disk (Low Memory)')
  logging.info('='*80)
  
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Load metadata
  logging.info('\n[1/4] Loading metadata...')
  metadata_path = os.path.join(FLAGS.averaged_dir, 'metadata.pkl')
  with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)
  
  activation_names = metadata['activation_names']
  logging.info(f'  Found {len(activation_names)} activation types')
  
  # Load class names
  labels_df = pd.read_csv(FLAGS.audioset_labels_csv)
  index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
  
  # Parse inputs
  logging.info('\n[2/4] Parsing class groupings...')
  
  if FLAGS.config:
    # Load from JSON config
    with open(FLAGS.config, 'r') as f:
      config = json.load(f)
    
    groupings = config.get('groupings', [])
    logging.info(f'  Loaded {len(groupings)} groupings from config')
  else:
    # Single grouping from command line
    class_indices = [int(x.strip()) for x in FLAGS.class_indices.split(',')]
    class_names = [index_to_name.get(idx, f'class_{idx}') for idx in class_indices]
    groupings = [{
        'name': FLAGS.output_name,
        'class_indices': class_indices,
        'class_names': class_names
    }]
    logging.info(f'  Combining {len(class_indices)} classes into "{FLAGS.output_name}"')
    logging.info(f'    Classes: {", ".join(class_names[:5])}' + 
                 (f' + {len(class_names)-5} more' if len(class_names) > 5 else ''))
  
  # Combine each grouping
  logging.info('\n[3/4] Combining classes...')
  
  results = []
  for i, grouping in enumerate(groupings):
    group_name = grouping['name']
    class_indices = grouping['class_indices']
    
    logging.info(f'  Grouping {i+1}/{len(groupings)}: {group_name} ({len(class_indices)} classes)')
    
    # Combine
    combined = combine_classes(class_indices, FLAGS.averaged_dir, activation_names)
    
    # Save
    group_output_dir = os.path.join(FLAGS.output_dir, group_name)
    save_combined_class(combined, group_name, group_output_dir, {
        'name': group_name,
        'source_classes': class_indices,
        'source_class_names': [index_to_name.get(idx, '') for idx in class_indices],
        'num_source_classes': len(class_indices),
        'num_activations': len(combined),
        'activation_names': list(combined.keys())
    })
    
    # Calculate total size
    total_size = sum(v.nbytes for v in combined.values())
    logging.info(f'    → Saved to {group_output_dir}/ ({total_size / (1024**2):.1f} MB)')
    
    results.append({
        'name': group_name,
        'num_classes': len(class_indices),
        'output_dir': group_output_dir
    })
  
  logging.info('\n[4/4] Summary...')
  summary_df = pd.DataFrame(results)
  summary_path = os.path.join(FLAGS.output_dir, 'summary.csv')
  summary_df.to_csv(summary_path, index=False)
  logging.info(f'  Saved summary to {summary_path}')
  
  logging.info('\n' + '='*80)
  logging.info('Class Combination Complete!')
  logging.info(f'Output directory: {FLAGS.output_dir}')
  logging.info('='*80)
  
  logging.info('\nTo load combined classes:')
  logging.info('  import numpy as np')
  logging.info('  # Load a combined class activation:')
  logging.info('  act = np.load("combined_classes/audio_content/audio_content_encoder_block_L0_rgb_output.npy")')
  logging.info('  # Compute distance between combined classes:')
  logging.info('  from scipy.spatial.distance import correlation')
  logging.info('  # Load two combined classes, flatten, and compute distance')


if __name__ == '__main__':
  app.run(main)
