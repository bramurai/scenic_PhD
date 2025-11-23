#!/usr/bin/env python3
"""Extract class-averaged activations from trained MBT model.

This script:
1. Loads a trained MBT checkpoint
2. Runs forward pass on all test samples
3. Accumulates activations per class (handles multi-label properly)
4. Saves only class-averaged activations (~26 GB instead of ~1.9 TB)

For multi-label samples:
  - Each sample contributes to the average of ALL its classes
  - Example: Sample with labels [Music, Speech] contributes to both Music and Speech averages

Storage comparison:
  - Per-sample storage: 3,853 samples × 0.5 GB = 1.9 TB
  - Class-averaged storage: 527 classes × ~50 MB = 26 GB (100x smaller!)

Usage:
  python extract_mbt_activations_class_averaged.py \
    --config=scenic/projects/mbt/configs/audioset/audioset_classification.py \
    --checkpoint_dir=mbt_base \
    --test_data_dir=Datasets/audioset_eval \
    --output_dir=audioset_class_averaged \
    --audioset_labels_csv=Video_csvs/audioset_labels.csv
"""

import os
import pickle
from typing import Dict, Any
from collections import defaultdict
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ml_collections
from flax.training import checkpoints
import tensorflow as tf

# Scenic imports
from scenic.projects.mbt import model as mbt_model
from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'Path to config file')
flags.DEFINE_string('checkpoint_dir', None, 'Directory containing checkpoint files')
flags.DEFINE_string('test_data_dir', None, 'Directory with test TFRecords')
flags.DEFINE_string('output_dir', 'audioset_class_averaged', 'Output directory')
flags.DEFINE_string('audioset_labels_csv', None, 'Path to audioset_labels.csv for class names')
flags.DEFINE_integer('num_samples', None, 'Number of samples to process (None = all)')
flags.DEFINE_bool('average_attention_heads', True, 'Average attention over heads to reduce size')
flags.DEFINE_integer('clear_cache_every', 10, 'Clear JAX cache every N samples')
flags.DEFINE_bool('save_attention', False, 'Save attention weights (increases storage significantly)')

flags.mark_flag_as_required('config')
flags.mark_flag_as_required('checkpoint_dir')
flags.mark_flag_as_required('test_data_dir')
flags.mark_flag_as_required('audioset_labels_csv')


def load_config(config_path: str) -> ml_collections.ConfigDict:
  """Load config from Python file."""
  import importlib.util
  spec = importlib.util.spec_from_file_location("config", config_path)
  config_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config_module)
  config = config_module.get_config()
  
  config.dataset_configs.base_dir = FLAGS.test_data_dir
  config.dataset_configs.tables = {
      'test': [os.path.join(FLAGS.test_data_dir, 'tar*', '*.tfrecord')],
  }
  
  return config


def create_test_dataset(config: ml_collections.ConfigDict):
  """Create test dataset iterator."""
  logging.info('Creating test dataset...')
  
  import glob
  import functools
  tfrecord_pattern = os.path.join(FLAGS.test_data_dir, '**', '*.tfrecord')
  tfrecord_files = glob.glob(tfrecord_pattern, recursive=True)
  
  if not tfrecord_files:
    raise ValueError(f'No TFRecords found in {FLAGS.test_data_dir}')
  
  logging.info(f'Found {len(tfrecord_files)} TFRecord files')
  
  tfrecord_files_relative = [os.path.relpath(f, FLAGS.test_data_dir) for f in tfrecord_files]
  
  # Determine number of samples
  num_samples = FLAGS.num_samples if FLAGS.num_samples else 100000  # Large number if None
  
  ds_factory = functools.partial(
      audiovisual_tfrecord_dataset.AVTFRecordDatasetFactory,
      base_dir=FLAGS.test_data_dir,
      tables={'test': tfrecord_files_relative},
      num_classes=config.dataset_configs.num_classes,
      examples_per_subset={'test': num_samples},
      num_groups=1,
      group_index=0
  )
  
  dataset, num_examples = audiovisual_tfrecord_dataset.load_split_from_dmvr(
      ds_factory=ds_factory,
      batch_size=1,
      subset='test',
      modalities=('spectrogram', 'rgb'),
      num_frames=config.dataset_configs.num_frames,
      stride=config.dataset_configs.stride,
      num_spec_frames=config.dataset_configs.num_spec_frames,
      spec_stride=config.dataset_configs.spec_stride,
      num_test_clips=1,
      min_resize=config.dataset_configs.min_resize,
      crop_size=config.dataset_configs.crop_size,
      spec_shape=config.dataset_configs.spec_shape,
      dataset_spec_mean=config.dataset_configs.get('spec_mean', 0.0),
      dataset_spec_stddev=config.dataset_configs.get('spec_stddev', 1.0),
      spec_augment=False,
      spec_augment_params=None,
      one_hot_label=True,
      zero_centering=True,
      augmentation_params=None,
  )
  
  from scenic.dataset_lib import dataset_utils
  return_as_dict = config.dataset_configs.get('return_as_dict', True)
  
  dataset_iter = iter(dataset)
  dataset_iter = map(dataset_utils.tf_to_numpy, dataset_iter)
  dataset_iter = map(
      functools.partial(
          audiovisual_tfrecord_dataset.map_keys, 
          modalities=('spectrogram', 'rgb'), 
          return_as_dict=return_as_dict),
      dataset_iter)
  
  logging.info(f'Dataset created with {num_examples} examples')
  return dataset_iter, num_examples


def load_checkpoint(config: ml_collections.ConfigDict, checkpoint_dir: str):
  """Load trained MBT checkpoint."""
  logging.info(f'Loading checkpoint from {checkpoint_dir}...')
  
  if os.path.isfile(checkpoint_dir):
    checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
  else:
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if not f.startswith('.') and os.path.isfile(os.path.join(checkpoint_dir, f))]
    
    if len(checkpoint_files) == 1:
      single_ckpt = os.path.join(checkpoint_dir, checkpoint_files[0])
      checkpoint_path = checkpoints.restore_checkpoint(single_ckpt, None)
    else:
      checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
  
  if checkpoint_path is None:
    raise ValueError(f'No checkpoint found in {checkpoint_dir}')
  
  if 'params' in checkpoint_path:
    params = checkpoint_path['params']
  elif 'optimizer' in checkpoint_path and 'target' in checkpoint_path['optimizer']:
    params = checkpoint_path['optimizer']['target']
  else:
    params = checkpoint_path
  
  model_cls = mbt_model.MBTMultilabelClassificationModel
  spec_time_dim = config.dataset_configs.num_spec_frames * config.dataset_configs.spec_shape[0]
  
  model_instance = model_cls(config, {
      'num_classes': config.dataset_configs.num_classes,
      'input_shape': {
          'rgb': (-1, config.dataset_configs.num_frames, 224, 224, 3),
          'spectrogram': (-1, spec_time_dim, config.dataset_configs.spec_shape[1], 3)
      },
      'input_dtype': jnp.float32,
      'target_is_onehot': True
  })
  
  model_state = {}
  rng = jax.random.PRNGKey(0)
  
  logging.info('Checkpoint loaded successfully')
  return model_instance, params, model_state, rng


def extract_with_intermediates(model_instance, params, model_state, inputs):
  """Extract activations using intermediate capture."""
  variables = {'params': params}
  if model_state:
    variables['batch_stats'] = model_state
  
  output, state = model_instance.flax_model.apply(
      variables,
      inputs,
      train=False,
      mutable=['intermediates'],
      capture_intermediates=True
  )
  
  activations = {}
  
  def extract_from_dict(d, prefix='', depth=0):
    """Recursively extract arrays from nested FrozenDicts."""
    has_items = hasattr(d, 'items')
    is_tuple = isinstance(d, tuple)
    
    if has_items or is_tuple:
      items = d.items() if has_items else enumerate(d)
      for key, value in items:
        new_prefix = f"{prefix}/{key}" if prefix else str(key)
        
        if hasattr(value, 'shape'):
          activations[new_prefix] = np.array(value)
        else:
          extract_from_dict(value, new_prefix, depth + 1)
  
  if 'intermediates' in state:
    extract_from_dict(state['intermediates'])
  
  return {
      'logits': np.array(output),
      'activations': activations
  }


def filter_essential_activations(activations: Dict) -> Dict:
  """Filter to keep only encoder block outputs and optionally attention.
  
  Returns dict with keys like:
    - encoder_block_L0_rgb_output
    - encoder_block_L0_audio_output
    - attention_weights_L0_rgb (if save_attention=True)
  """
  essential = {}
  
  # Extract encoder block outputs
  for key, value in activations.items():
    if 'encoderblock_' in key and 'MlpBlock_0/__call__/0' in key and 'Transformer' in key:
      parts = key.split('/')
      for part in parts:
        if part.startswith('encoderblock_'):
          if '_spectrogram' in part:
            layer_num = part.replace('encoderblock_', '').replace('_spectrogram', '')
            name = f'encoder_block_L{layer_num}_audio_output'
          else:
            layer_num = part.replace('encoderblock_', '')
            name = f'encoder_block_L{layer_num}_rgb_output'
          essential[name] = value
          break
  
  # Optionally compute attention weights
  if FLAGS.save_attention:
    attention_layers = {}
    
    for key in activations.keys():
      if 'MultiHeadDotProductAttention' in key and 'encoderblock_' in key and 'Transformer' in key:
        parts = key.split('/')
        for part in parts:
          if part.startswith('encoderblock_'):
            base_name = part
            if base_name not in attention_layers:
              attention_layers[base_name] = {}
            
            if '/query/__call__/0' in key:
              attention_layers[base_name]['query'] = activations[key]
            elif '/key/__call__/0' in key:
              attention_layers[base_name]['key'] = activations[key]
            break
    
    # Compute attention from Q and K
    for layer_name, qk_dict in attention_layers.items():
      if 'query' in qk_dict and 'key' in qk_dict:
        Q = qk_dict['query']
        K = qk_dict['key']
        
        Q_t = np.transpose(Q, (0, 2, 1, 3))
        K_t = np.transpose(K, (0, 2, 3, 1))
        
        head_dim = Q.shape[-1]
        scores = np.matmul(Q_t, K_t) / np.sqrt(head_dim)
        
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        if FLAGS.average_attention_heads:
          attention = np.mean(attention, axis=1)
          attention = attention[0]
        
        if 'spectrogram' in layer_name:
          layer_num = layer_name.split('_')[1]
          name = f'attention_weights_L{layer_num}_audio'
        else:
          layer_num = layer_name.split('_')[1]
          name = f'attention_weights_L{layer_num}_rgb'
        
        essential[name] = attention
  
  return essential


class ClassAccumulator:
  """Accumulates activations per class for averaging."""
  
  def __init__(self, num_classes: int):
    self.num_classes = num_classes
    # Store sum and count for each class
    self.sums = defaultdict(lambda: defaultdict(lambda: None))
    self.counts = defaultdict(int)
  
  def add_sample(self, activations: Dict, label: np.ndarray):
    """Add a sample's activations to all its classes.
    
    Args:
      activations: Dict of activation arrays
      label: Multi-hot label vector (shape: num_classes)
    """
    # Find which classes this sample belongs to
    active_classes = np.where(label > 0)[0]
    
    for class_idx in active_classes:
      class_idx = int(class_idx)
      self.counts[class_idx] += 1
      
      # Add activations to this class's sum
      for act_name, act_value in activations.items():
        if self.sums[class_idx][act_name] is None:
          self.sums[class_idx][act_name] = np.zeros_like(act_value)
        
        self.sums[class_idx][act_name] += act_value
  
  def compute_averages(self) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute average activations for each class.
    
    Returns:
      Dict mapping class_idx -> {activation_name: averaged_array}
    """
    averages = {}
    
    for class_idx in self.sums.keys():
      count = self.counts[class_idx]
      if count == 0:
        continue
      
      averages[class_idx] = {}
      for act_name, act_sum in self.sums[class_idx].items():
        averages[class_idx][act_name] = act_sum / count
    
    return averages
  
  def get_stats(self) -> Dict:
    """Get statistics about accumulation."""
    return {
        'num_classes_with_samples': len(self.counts),
        'samples_per_class': dict(self.counts),
        'min_samples': min(self.counts.values()) if self.counts else 0,
        'max_samples': max(self.counts.values()) if self.counts else 0,
        'mean_samples': np.mean(list(self.counts.values())) if self.counts else 0
    }


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('MBT Class-Averaged Activation Extraction')
  logging.info('='*80)
  
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Load AudioSet labels
  logging.info('\n[1/5] Loading AudioSet labels...')
  labels_df = pd.read_csv(FLAGS.audioset_labels_csv)
  index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
  index_to_mid = dict(zip(labels_df['index'], labels_df['mid']))
  num_classes = len(labels_df)
  logging.info(f'Loaded {num_classes} AudioSet class labels')
  
  # Load config
  logging.info('\n[2/5] Loading configuration...')
  config = load_config(FLAGS.config)
  
  # Load checkpoint
  logging.info('\n[3/5] Loading checkpoint...')
  model_instance, params, model_state, rng = load_checkpoint(config, FLAGS.checkpoint_dir)
  
  # Create dataset
  logging.info('\n[4/5] Loading test data...')
  dataset, num_examples = create_test_dataset(config)
  num_to_process = min(FLAGS.num_samples, num_examples) if FLAGS.num_samples else num_examples
  
  # Initialize accumulator
  logging.info(f'\n[5/5] Processing {num_to_process} samples and accumulating by class...')
  accumulator = ClassAccumulator(num_classes)
  
  processed_count = 0
  
  for sample_idx, batch in enumerate(dataset):
    if sample_idx >= num_to_process:
      break
    
    if processed_count % 100 == 0:
      logging.info(f'  Processed {processed_count}/{num_to_process} samples...')
    
    try:
      inputs = batch['inputs']
      labels = batch['label'].squeeze()  # Multi-hot label vector
      
      # Extract activations
      result = extract_with_intermediates(model_instance, params, model_state, inputs)
      
      # Filter to essential activations
      essential = filter_essential_activations(result['activations'])
      
      # Remove batch dimension (we process one at a time)
      for key in essential:
        if essential[key].ndim > 0 and essential[key].shape[0] == 1:
          essential[key] = essential[key][0]
      
      # Add to accumulator
      accumulator.add_sample(essential, labels)
      
      processed_count += 1
      
      # Log first sample info
      if sample_idx == 0:
        logging.info(f'\nFirst sample activations:')
        for name, value in essential.items():
          logging.info(f'  {name}: shape {value.shape}, size {value.nbytes / 1024**2:.1f} MB')
        
        active_classes = np.where(labels > 0)[0]
        logging.info(f'\nFirst sample has {len(active_classes)} active classes:')
        for class_idx in active_classes[:5]:  # Show first 5
          logging.info(f'  - {index_to_name[class_idx]}')
      
      # Clear JAX cache periodically
      if processed_count % FLAGS.clear_cache_every == 0:
        jax.clear_caches()
        import gc
        gc.collect()
    
    except Exception as e:
      logging.error(f'ERROR processing sample {sample_idx}: {e}')
      import traceback
      logging.error(traceback.format_exc())
      continue
  
  # Compute averages
  logging.info('\nComputing class averages...')
  class_averages = accumulator.compute_averages()
  stats = accumulator.get_stats()
  
  logging.info(f'\nAccumulation Statistics:')
  logging.info(f'  Classes with samples: {stats["num_classes_with_samples"]}/{num_classes}')
  logging.info(f'  Samples per class - min: {stats["min_samples"]}, max: {stats["max_samples"]}, mean: {stats["mean_samples"]:.1f}')
  
  # Save class-averaged activations
  logging.info('\nSaving class-averaged activations...')
  output_path = os.path.join(FLAGS.output_dir, 'class_averaged_activations.npz')
  
  # Prepare save dict with flattened keys: class_0_encoder_block_L0_rgb_output, etc.
  save_dict = {}
  total_size = 0
  
  for class_idx, activations in class_averages.items():
    class_name = index_to_name.get(class_idx, f'class_{class_idx}')
    
    for act_name, act_value in activations.items():
      key = f'class_{class_idx}_{act_name}'
      save_dict[key] = act_value
      total_size += act_value.nbytes
  
  # Add metadata
  save_dict['class_names'] = np.array([index_to_name.get(i, '') for i in range(num_classes)], dtype=object)
  save_dict['class_mids'] = np.array([index_to_mid.get(i, '') for i in range(num_classes)], dtype=object)
  save_dict['samples_per_class'] = np.array([stats['samples_per_class'].get(i, 0) for i in range(num_classes)])
  save_dict['num_classes'] = num_classes
  save_dict['num_samples_processed'] = processed_count
  
  logging.info(f'  Total arrays to save: {len([k for k in save_dict.keys() if k.startswith("class_")])}')
  logging.info(f'  Total size: {total_size / (1024**3):.2f} GB')
  
  np.savez_compressed(output_path, **save_dict)
  
  # Save detailed class statistics
  class_stats = []
  for class_idx in range(num_classes):
    count = stats['samples_per_class'].get(class_idx, 0)
    class_stats.append({
        'index': class_idx,
        'mid': index_to_mid.get(class_idx, ''),
        'display_name': index_to_name.get(class_idx, ''),
        'num_samples': count
    })
  
  stats_df = pd.DataFrame(class_stats)
  stats_path = os.path.join(FLAGS.output_dir, 'class_statistics.csv')
  stats_df.to_csv(stats_path, index=False)
  
  # Save metadata
  metadata = {
      'checkpoint_dir': FLAGS.checkpoint_dir,
      'test_data_dir': FLAGS.test_data_dir,
      'num_samples_processed': processed_count,
      'num_classes': num_classes,
      'config': config.to_dict(),
      'statistics': stats
  }
  metadata_path = os.path.join(FLAGS.output_dir, 'metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
  
  logging.info('\n' + '='*80)
  logging.info('Class-Averaged Extraction Complete!')
  logging.info(f'Processed {processed_count} samples')
  logging.info(f'Computed averages for {stats["num_classes_with_samples"]} classes')
  logging.info(f'Total storage: {total_size / (1024**3):.2f} GB')
  logging.info(f'Output saved to: {output_path}')
  logging.info('='*80)
  
  # Print usage instructions
  logging.info('\nTo load class-averaged activations:')
  logging.info('  import numpy as np')
  logging.info(f'  data = np.load("{output_path}")')
  logging.info('  # Get activation for class 137 (Music), layer 0, RGB:')
  logging.info('  music_L0_rgb = data["class_137_encoder_block_L0_rgb_output"]')
  logging.info('  # Get class names:')
  logging.info('  class_names = data["class_names"]')
  logging.info('  samples_per_class = data["samples_per_class"]')
  
  # Show top 10 classes by sample count
  logging.info('\nTop 10 classes by sample count:')
  top_classes = sorted(stats['samples_per_class'].items(), key=lambda x: x[1], reverse=True)[:10]
  for class_idx, count in top_classes:
    logging.info(f'  {index_to_name[class_idx]}: {count} samples')


if __name__ == '__main__':
  app.run(main)
