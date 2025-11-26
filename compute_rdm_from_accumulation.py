#!/usr/bin/env python3
"""Compute RDMs directly from .accumulation files without loading all data into RAM.

This script:
1. Reads class-averaged activations from individual .npy files (streaming)
2. Computes pairwise distances between classes (one pair at a time)
3. Builds RDM incrementally without ever loading all data simultaneously
4. Saves RDM matrix to disk

Key advantage: Works with ~88 GB of activation data on a 62 GB system
by processing one class pair at a time.

Usage:
  python compute_rdm_from_accumulation.py \
    --accumulation_dir=audioset_analysis_AV/.accumulation \
    --checkpoint_path=audioset_analysis_AV/checkpoint.pkl \
    --audioset_labels_csv=Video_csvs/audioset_labels.csv \
    --output_dir=RDM_from_accumulation \
    --distance_metric=correlation
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from absl import app, flags, logging
from scipy.spatial.distance import pdist, squareform, correlation
from scipy import stats

FLAGS = flags.FLAGS

flags.DEFINE_string('accumulation_dir', None, 'Path to .accumulation directory')
flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint.pkl with metadata')
flags.DEFINE_string('audioset_labels_csv', None, 'Path to audioset_labels.csv')
flags.DEFINE_string('output_dir', 'RDM_from_accumulation', 'Output directory')
flags.DEFINE_string('distance_metric', 'correlation', 'Distance metric: correlation, euclidean, cosine')
flags.DEFINE_integer('batch_size', 50, 'Number of classes to hold in memory at once')

flags.mark_flag_as_required('accumulation_dir')
flags.mark_flag_as_required('checkpoint_path')
flags.mark_flag_as_required('audioset_labels_csv')


def load_checkpoint_metadata(checkpoint_path):
  """Load metadata from checkpoint without loading large data."""
  with open(checkpoint_path, 'rb') as f:
    data = pickle.load(f)
  
  return {
      'counts': data['counts'],
      'num_classes': data['num_classes'],
      'activation_names': data.get('activation_names', []),
      'processed_count': data.get('processed_count', 0)
  }


def load_class_activation(class_idx, act_name, accumulation_dir, checkpoint_data):
  """Load a single class activation from disk and compute average.
  
  Returns:
    averaged_activation: np.array of shape matching the activation
    count: number of samples that went into this class
  """
  count = checkpoint_data['counts'].get(class_idx, 0)
  if count == 0:
    return None, 0
  
  sum_path = os.path.join(accumulation_dir, f'class_{class_idx}_{act_name}.npy')
  if not os.path.exists(sum_path):
    return None, 0
  
  activation_sum = np.load(sum_path)
  averaged = activation_sum / count
  
  return averaged, count


def flatten_activations(class_idx, accumulation_dir, checkpoint_data):
  """Flatten all activations for a class into a single 1D vector.
  
  Returns:
    flattened_vector: 1D np.array
    num_elements: total number of elements
  """
  activation_names = checkpoint_data['activation_names']
  vectors = []
  
  for act_name in activation_names:
    avg_act, _ = load_class_activation(class_idx, act_name, accumulation_dir, checkpoint_data)
    if avg_act is not None:
      vectors.append(avg_act.flatten())
  
  if not vectors:
    return None
  
  return np.concatenate(vectors)


def compute_rdm_streaming(accumulation_dir, checkpoint_data, distance_metric='correlation',
                         batch_size=50):
  """Compute RDM by processing classes in batches to avoid loading everything at once.
  
  Returns:
    rdm_matrix: full RDM matrix (n_classes x n_classes)
    class_indices: indices of classes that had samples
  """
  num_classes = checkpoint_data['num_classes']
  counts = checkpoint_data['counts']
  
  # Get list of classes that have samples
  class_indices = sorted([c for c in counts.keys() if counts[c] > 0])
  n_classes_with_samples = len(class_indices)
  
  logging.info(f'Computing RDM for {n_classes_with_samples} classes with samples')
  logging.info(f'Using batch size: {batch_size} (keeps ~{batch_size * 50:.0f} MB in RAM)')
  
  # Initialize RDM matrix
  rdm_matrix = np.zeros((n_classes_with_samples, n_classes_with_samples), dtype=np.float32)
  
  # Process in batches to keep memory low
  for batch_start in range(0, n_classes_with_samples, batch_size):
    batch_end = min(batch_start + batch_size, n_classes_with_samples)
    batch_indices = class_indices[batch_start:batch_end]
    
    # Load activations for this batch
    logging.info(f'Loading batch {batch_start//batch_size + 1}: classes {batch_start}-{batch_end}/{n_classes_with_samples}')
    batch_activations = {}
    
    for class_idx in batch_indices:
      flattened = flatten_activations(class_idx, accumulation_dir, checkpoint_data)
      if flattened is not None:
        batch_activations[class_idx] = flattened
    
    logging.info(f'  Loaded {len(batch_activations)} classes, computing distances...')
    
    # Compute distances: batch_class x all_classes
    for i, class_i_idx in enumerate(batch_indices):
      act_i = batch_activations.get(class_i_idx)
      if act_i is None:
        continue
      
      row_idx = class_indices.index(class_i_idx)
      
      # Distance to all classes (only compute upper triangle for efficiency)
      for j, class_j_idx in enumerate(class_indices):
        if j < row_idx:
          # Use symmetry: copy from lower triangle
          col_idx = class_indices.index(class_j_idx)
          rdm_matrix[row_idx, col_idx] = rdm_matrix[col_idx, row_idx]
        else:
          # Compute distance
          flattened_j = flatten_activations(class_j_idx, accumulation_dir, checkpoint_data)
          if flattened_j is None:
            continue
          
          if distance_metric == 'correlation':
            # Correlation distance: 1 - correlation
            dist = 1.0 - np.corrcoef(act_i, flattened_j)[0, 1]
          elif distance_metric == 'euclidean':
            dist = np.linalg.norm(act_i - flattened_j)
          elif distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            dot_product = np.dot(act_i, flattened_j)
            norm_i = np.linalg.norm(act_i)
            norm_j = np.linalg.norm(flattened_j)
            if norm_i > 0 and norm_j > 0:
              dist = 1.0 - (dot_product / (norm_i * norm_j))
            else:
              dist = 1.0
          else:
            raise ValueError(f'Unknown metric: {distance_metric}')
          
          col_idx = class_indices.index(class_j_idx)
          rdm_matrix[row_idx, col_idx] = dist
    
    # Clear batch to free memory
    del batch_activations
    import gc
    gc.collect()
  
  return rdm_matrix, class_indices


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('Computing RDM from Accumulation Files (Streaming)')
  logging.info('='*80)
  
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Load metadata
  logging.info('\n[1/4] Loading checkpoint metadata...')
  checkpoint_data = load_checkpoint_metadata(FLAGS.checkpoint_path)
  logging.info(f'  Classes with samples: {len(checkpoint_data["counts"])}')
  logging.info(f'  Total samples processed: {checkpoint_data["processed_count"]}')
  logging.info(f'  Activation names: {len(checkpoint_data["activation_names"])}')
  
  # Load class names
  logging.info('\n[2/4] Loading AudioSet labels...')
  labels_df = pd.read_csv(FLAGS.audioset_labels_csv)
  index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
  index_to_mid = dict(zip(labels_df['index'], labels_df['mid']))
  
  # Compute RDM
  logging.info('\n[3/4] Computing RDM (streaming from disk)...')
  rdm_matrix, class_indices = compute_rdm_streaming(
      FLAGS.accumulation_dir,
      checkpoint_data,
      distance_metric=FLAGS.distance_metric,
      batch_size=FLAGS.batch_size
  )
  
  logging.info(f'  RDM shape: {rdm_matrix.shape}')
  logging.info(f'  RDM min: {rdm_matrix.min():.4f}, max: {rdm_matrix.max():.4f}, mean: {rdm_matrix.mean():.4f}')
  
  # Verify RDM is symmetric (should be for correlation/euclidean)
  if FLAGS.distance_metric in ['correlation', 'euclidean', 'cosine']:
    asymmetry = np.max(np.abs(rdm_matrix - rdm_matrix.T))
    logging.info(f'  RDM asymmetry: {asymmetry:.2e} (should be near 0)')
  
  # Save RDM
  logging.info('\n[4/4] Saving results...')
  
  # Save as NPZ with metadata
  output_npz = os.path.join(FLAGS.output_dir, 'rdm_matrix.npz')
  class_names = np.array([index_to_name.get(idx, '') for idx in class_indices], dtype=object)
  class_mids = np.array([index_to_mid.get(idx, '') for idx in class_indices], dtype=object)
  
  np.savez_compressed(
      output_npz,
      rdm_matrix=rdm_matrix,
      class_indices=np.array(class_indices),
      class_names=class_names,
      class_mids=class_mids,
      distance_metric=FLAGS.distance_metric,
      num_classes_in_rdm=len(class_indices)
  )
  
  logging.info(f'  Saved RDM to {output_npz}')
  
  # Save as CSV for easy inspection
  output_csv = os.path.join(FLAGS.output_dir, 'rdm_matrix.csv')
  rdm_df = pd.DataFrame(rdm_matrix, columns=class_names, index=class_names)
  rdm_df.to_csv(output_csv)
  logging.info(f'  Saved RDM CSV to {output_csv}')
  
  # Save class information
  class_info = pd.DataFrame({
      'index': class_indices,
      'mid': class_mids,
      'display_name': class_names,
      'sample_count': [checkpoint_data['counts'][idx] for idx in class_indices]
  })
  class_info_path = os.path.join(FLAGS.output_dir, 'class_info.csv')
  class_info.to_csv(class_info_path, index=False)
  logging.info(f'  Saved class info to {class_info_path}')
  
  logging.info('\n' + '='*80)
  logging.info('RDM Computation Complete!')
  logging.info(f'RDM shape: {rdm_matrix.shape}')
  logging.info(f'Distance metric: {FLAGS.distance_metric}')
  logging.info(f'Output directory: {FLAGS.output_dir}')
  logging.info('='*80)
  
  logging.info('\nTo load and inspect RDM:')
  logging.info('  import numpy as np')
  logging.info(f'  data = np.load("{output_npz}")')
  logging.info('  rdm = data["rdm_matrix"]')
  logging.info('  class_names = data["class_names"]')
  logging.info('  # Plot RDM:')
  logging.info('  import matplotlib.pyplot as plt')
  logging.info('  plt.imshow(rdm, cmap="viridis")')
  logging.info('  plt.colorbar(label="Distance")')
  logging.info('  plt.title("AudioSet Class RDM")')
  logging.info('  plt.tight_layout()')
  logging.info('  plt.savefig("rdm_heatmap.png", dpi=150, bbox_inches="tight")')


if __name__ == '__main__':
  app.run(main)
