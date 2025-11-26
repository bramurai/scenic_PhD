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
import time
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
flags.DEFINE_integer('batch_size', 4, 'Batch size for processing (higher = faster but more memory)')
flags.DEFINE_bool('average_attention_heads', True, 'Average attention over heads to reduce size')
flags.DEFINE_integer('clear_cache_every', 1, 'Clear JAX cache every N samples')
flags.DEFINE_bool('save_attention', False, 'Save attention weights (increases storage significantly)')
flags.DEFINE_integer('checkpoint_every', 1, 'Save intermediate checkpoint every N batches (0 = disable)')
flags.DEFINE_bool('resume_from_checkpoint', True, 'Resume from checkpoint if it exists')

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


def read_labels_from_tfrecord(tfrecord_path: str) -> list:
  """Read ALL multi-hot labels from a TFRecord file.
  
  Args:
    tfrecord_path: Path to TFRecord file
    
  Returns:
    List of multi-hot label vectors, one per sample in the file
  """
  labels = []
  
  # Read ALL records from this file (not just the first one)
  for raw_record in tf.data.TFRecordDataset([tfrecord_path]):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    # Read multi-hot label from context
    if 'clip/label/multi_hot' in example.context.feature:
      label_floats = example.context.feature['clip/label/multi_hot'].float_list.value
      labels.append(np.array(label_floats, dtype=np.float32))
    else:
      raise ValueError(f'No clip/label/multi_hot found in {tfrecord_path}')
  
  return labels


def create_test_dataset(config: ml_collections.ConfigDict):
  """Create test dataset iterator with labels read separately."""
  logging.info('Creating test dataset...')
  
  import glob
  import functools
  tfrecord_pattern = os.path.join(FLAGS.test_data_dir, '**', '*.tfrecord')
  tfrecord_files = glob.glob(tfrecord_pattern, recursive=True)
  
  if not tfrecord_files:
    raise ValueError(f'No TFRecords found in {FLAGS.test_data_dir}')
  
  logging.info(f'Found {len(tfrecord_files)} TFRecord files')
  
  # Sort for deterministic order
  tfrecord_files.sort()
  
  tfrecord_files_relative = [os.path.relpath(f, FLAGS.test_data_dir) for f in tfrecord_files]
  
  # Determine number of samples
  num_samples = FLAGS.num_samples if FLAGS.num_samples else 100000  # Large number if None
  
  # Create dataset WITHOUT labels (we'll read them separately)
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
      batch_size=FLAGS.batch_size,  # Use configurable batch size
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
  
  # Return both dataset iterator and tfrecord file paths for label reading
  return dataset_iter, num_examples, tfrecord_files


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


# JIT-compile the forward pass for speed
@jax.jit
def forward_pass_jit(params, model_state, inputs):
  """JIT-compiled forward pass (faster after first compilation)."""
  variables = {'params': params}
  if model_state:
    variables['batch_stats'] = model_state
  
  # Note: We can't capture intermediates in JIT mode, so this is just for the forward pass
  # We'll use the non-JIT version when we need intermediates
  return None  # Placeholder - we'll use extract_with_intermediates instead


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


def save_checkpoint(accumulator, processed_count, output_dir, checkpoint_name='checkpoint.pkl'):
  """Save intermediate checkpoint with only counts (activation sums are already on disk)."""
  checkpoint_path = os.path.join(output_dir, checkpoint_name)
  
  # Only save counts and processed count (sums are stored on disk in .accumulation/)
  checkpoint_data = {
      'processed_count': processed_count,
      'counts': dict(accumulator.counts),
      'num_classes': accumulator.num_classes,
      'activation_names': accumulator.activation_names
  }
  with open(checkpoint_path, 'wb') as f:
    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
  
  # Immediately delete temporary copy to free memory
  del checkpoint_data
  
  # Sync to disk and drop OS caches
  os.sync()
  try:
    # Drop OS page cache, dentries, inodes
    with open('/proc/sys/vm/drop_caches', 'w') as f:
      f.write('3')  # 3 = drop everything (pagecache, dentries, inodes)
    logging.info(f'  Saved checkpoint to {checkpoint_path} and dropped OS caches')
  except PermissionError:
    logging.info(f'  Saved checkpoint to {checkpoint_path} (no permission to drop OS caches)')
  except Exception as e:
    logging.info(f'  Saved checkpoint to {checkpoint_path} (could not drop OS caches: {e})')


def load_checkpoint_if_exists(output_dir, num_classes, checkpoint_name='checkpoint.pkl'):
  """Load checkpoint if it exists. Activation sums remain on disk.
  
  Returns:
      processed_count or 0 if no checkpoint
  """
  checkpoint_path = os.path.join(output_dir, checkpoint_name)
  
  if not os.path.exists(checkpoint_path):
    return 0
  
  try:
    with open(checkpoint_path, 'rb') as f:
      checkpoint_data = pickle.load(f)
    
    processed_count = checkpoint_data['processed_count']
    
    logging.info(f'Loaded checkpoint from {checkpoint_path}')
    logging.info(f'  Resuming from sample {processed_count}')
    logging.info(f'  Already accumulated {len(checkpoint_data["counts"])} classes')
    logging.info(f'  Activation sums remain on disk in .accumulation/')
    
    return processed_count
  except Exception as e:
    logging.warning(f'Failed to load checkpoint: {e}. Starting from scratch.')
    return 0


class ClassAccumulator:
  """Accumulates activations per class for averaging - stores on disk to save RAM.
  
  Instead of keeping all sums in memory, we store them in individual numpy files.
  This allows the accumulator to scale to any size without RAM constraints.
  """
  
  def __init__(self, num_classes: int, output_dir: str):
    self.num_classes = num_classes
    self.output_dir = output_dir
    self.accumulation_dir = os.path.join(output_dir, '.accumulation')
    os.makedirs(self.accumulation_dir, exist_ok=True)
    
    # Keep only counts in RAM (very small - just 527 integers)
    self.counts = defaultdict(int)
    
    # Track which activation names exist (from first batch)
    self.activation_names = None
  
  def _get_sum_path(self, class_idx: int, act_name: str) -> str:
    """Get the file path for a class activation sum."""
    return os.path.join(self.accumulation_dir, f'class_{class_idx}_{act_name}.npy')
  
  def add_sample(self, activations: Dict, labels: np.ndarray):
    """Add sample(s) activations to all their classes.
    
    Args:
      activations: Dict of activation arrays
      labels: Multi-hot label vector(s) - shape: (num_classes,) or (batch, num_classes)
    """
    # Store activation names from first call
    if self.activation_names is None:
      self.activation_names = list(activations.keys())
    
    # Handle both single sample and batched inputs
    if labels.ndim == 1:
      # Single sample
      active_classes = np.where(labels > 0)[0]
      
      for class_idx in active_classes:
        class_idx = int(class_idx)
        self.counts[class_idx] += 1
        
        for act_name, act_value in activations.items():
          sum_path = self._get_sum_path(class_idx, act_name)
          act_value = np.asarray(act_value, dtype=np.float32)
          
          if os.path.exists(sum_path):
            print("skipped")
            # Load existing sum, add to it, save back
            #existing_sum = np.load(sum_path)
            #existing_sum[:] += act_value
            #np.save(sum_path, existing_sum)
          else:
            # First time seeing this class - save initial sum
            print("skipped")
            #np.save(sum_path, np.array(act_value, copy=True, dtype=np.float32))
    
    else:
      # Batched samples - labels shape: (batch, num_classes)
      batch_size = labels.shape[0]
      
      for batch_idx in range(batch_size):
        label = labels[batch_idx]
        active_classes = np.where(label > 0)[0]
        
        for class_idx in active_classes:
          class_idx = int(class_idx)
          self.counts[class_idx] += 1
          
          for act_name, act_value in activations.items():
            sample_activation = np.asarray(act_value[batch_idx], dtype=np.float32)
            sum_path = self._get_sum_path(class_idx, act_name)
            
            if os.path.exists(sum_path):
              print("skipped")
              # Load existing sum, add to it, save back
              # existing_sum = np.load(sum_path)
              # existing_sum[:] += sample_activation
              # np.save(sum_path, existing_sum)
            else:
              print("skipped")
              # First time seeing this class - save initial sum
              #np.save(sum_path, np.array(sample_activation, copy=True, dtype=np.float32))
  
  def compute_averages(self) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute average activations for each class by loading from disk.
    
    Returns:
      Dict mapping class_idx -> {activation_name: averaged_array}
    """
    averages = {}
    
    for class_idx in self.counts.keys():
      count = self.counts[class_idx]
      if count == 0:
        continue
      
      averages[class_idx] = {}
      for act_name in self.activation_names:
        sum_path = self._get_sum_path(class_idx, act_name)
        if os.path.exists(sum_path):
          act_sum = np.load(sum_path)
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
  
  def cleanup_accumulation_files(self):
    """Delete temporary accumulation files after saving final output.
    
    NOTE: Currently disabled to preserve .accumulation files for inspection.
    """
    # Accumulation files now preserved for debugging/inspection
    logging.info('Keeping .accumulation directory (cleanup disabled)')
    # import shutil
    # if os.path.exists(self.accumulation_dir):
    #   shutil.rmtree(self.accumulation_dir)
    #   logging.info('Removed temporary accumulation files')


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('MBT Class-Averaged Activation Extraction')
  logging.info('='*80)
  
  # Clear memory caches on startup
  logging.info('Clearing memory caches on startup...')
  import gc
  import subprocess
  gc.collect()
  gc.collect()
  try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    logging.info('Python memory cache cleared successfully')
  except Exception as e:
    logging.warning(f'Could not clear Python memory cache: {e}')
  
  # Clear OS-level caches (pagecache, dentries, inodes)
  logging.info('Clearing OS-level caches...')
  try:
    os.sync()
    subprocess.run(['sync'], check=True)
    # Try with sudo
    result = subprocess.run(['sudo', 'tee', '/proc/sys/vm/drop_caches'], 
                          input=b'3', capture_output=True)
    if result.returncode == 0:
      logging.info('OS caches cleared successfully')
    else:
      logging.warning('Could not clear OS caches (no sudo access)')
  except Exception as e:
    logging.warning(f'Could not clear OS caches: {e}')
  
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
  dataset, num_examples, tfrecord_files = create_test_dataset(config)
  num_to_process = min(FLAGS.num_samples, num_examples) if FLAGS.num_samples else num_examples
  
  # Create label iterator (stream instead of pre-loading to save RAM)
  logging.info('Creating label iterator from TFRecords...')
  
  def create_label_iterator(tfrecord_files):
    """Generator that yields labels on-demand instead of loading all into RAM."""
    for i, tfr_path in enumerate(tfrecord_files):
      if (i + 1) % 10 == 0 or i == 0:
        logging.info(f'  Streaming labels from file {i+1}/{len(tfrecord_files)}...')
      try:
        file_labels = read_labels_from_tfrecord(tfr_path)
        if i == 0:
          logging.info(f'    First file contains {len(file_labels)} samples')
        for label in file_labels:
          yield label
      except Exception as e:
        logging.warning(f'Failed to read labels from {tfr_path}: {e}')
        continue
  
  # Create iterator instead of list
  label_iterator = create_label_iterator(tfrecord_files)
  
  logging.info(f'Label iterator created (streaming from {len(tfrecord_files)} TFRecord files)')
  logging.info(f'Will process up to {num_to_process} samples from dataset')
  logging.info('Note: Using label streaming to minimize RAM usage')
  
  # Initialize accumulator
  logging.info(f'\n[5/5] Processing {num_to_process} samples and accumulating by class...')
  logging.info(f'Using batch size: {FLAGS.batch_size}')
  
  # Initialize accumulator with disk storage
  accumulator = ClassAccumulator(num_classes, FLAGS.output_dir)
  
  # Try to resume from checkpoint
  if FLAGS.resume_from_checkpoint:
    processed_count = load_checkpoint_if_exists(FLAGS.output_dir, num_classes)
    if processed_count == 0:
      logging.info('No checkpoint found, starting from beginning')
    else:
      # Load counts and activation names from checkpoint
      checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint.pkl')
      with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
      accumulator.counts = defaultdict(int, checkpoint_data['counts'])
      accumulator.activation_names = checkpoint_data.get('activation_names', None)
      
      # Skip ahead in label iterator to resume point
      logging.info(f'Skipping {processed_count} labels to resume from checkpoint...')
      for _ in range(processed_count):
        try:
          next(label_iterator)
        except StopIteration:
          logging.error('Label iterator exhausted while trying to skip to checkpoint position!')
          break
      logging.info(f'Resumed from checkpoint at sample {processed_count}')
  else:
    processed_count = 0
    logging.info('Checkpoint resume disabled, starting from beginning')
  
  # Calculate starting batch count from processed samples
  batch_count = processed_count // FLAGS.batch_size
  start_time = time.time()
  
  for batch_idx, batch in enumerate(dataset):
    if processed_count >= num_to_process:
      break
    
    batch_start_time = time.time()
    
    current_batch_size = batch['inputs']['rgb'].shape[0] if 'rgb' in batch['inputs'] else batch['inputs']['spectrogram'].shape[0]
    
    # Calculate which sample indices this batch contains
    batch_start_idx = batch_idx * FLAGS.batch_size
    batch_end_idx = batch_start_idx + current_batch_size
    
    # Skip if we've already processed this batch (resuming from checkpoint)
    if batch_end_idx <= processed_count:
      continue
    
    # Partial skip: some samples in this batch were already processed
    if batch_start_idx < processed_count < batch_end_idx:
      skip_samples = processed_count - batch_start_idx
      for key in batch['inputs']:
        batch['inputs'][key] = batch['inputs'][key][skip_samples:]
      current_batch_size -= skip_samples
      batch_start_idx = processed_count
    
    if processed_count + current_batch_size > num_to_process:
      # Trim the last batch if it exceeds num_to_process
      samples_to_take = num_to_process - processed_count
      for key in batch['inputs']:
        batch['inputs'][key] = batch['inputs'][key][:samples_to_take]
      current_batch_size = samples_to_take
    
    # Log progress for every batch with timing
    elapsed = time.time() - start_time
    samples_per_sec = processed_count / elapsed if elapsed > 0 else 0
    eta_seconds = (num_to_process - processed_count) / samples_per_sec if samples_per_sec > 0 else 0
    eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m" if eta_seconds > 0 else "calculating..."
    
    logging.info(f'Batch {batch_count}: Processing samples {processed_count}-{processed_count + current_batch_size}/{num_to_process} | Speed: {samples_per_sec:.2f} samples/sec | ETA: {eta_str}')
    
    try:
      inputs = batch['inputs']
      
      # Get labels for this batch from iterator (streaming, not pre-loaded)
      batch_labels = []
      for i in range(current_batch_size):
        try:
          label = next(label_iterator)
          batch_labels.append(label)
        except StopIteration:
          # Ran out of labels - shouldn't happen but handle gracefully
          logging.warning(f'Label iterator exhausted at sample {processed_count + i}. Using zero vector.')
          batch_labels.append(np.zeros(num_classes, dtype=np.float32))
      
      batch_labels = np.array(batch_labels)
      
      # Extract activations for the batch
      result = extract_with_intermediates(model_instance, params, model_state, inputs)
      
      # Filter to essential activations (convert JAX arrays to numpy immediately)
      essential_jax = filter_essential_activations(result['activations'])
      
      # Convert JAX DeviceArrays to numpy arrays to free device memory
      essential = {}
      for key, value in essential_jax.items():
        essential[key] = np.array(value)  # Force copy to host memory
      del essential_jax
      
      # Immediately delete the full activations dict to free memory
      del result
      
      # Log first batch info BEFORE deleting
      if batch_idx == 0:
        logging.info('\nFirst batch activations:')
        for name, value in essential.items():
          logging.info(f'  {name}: shape {value.shape}, size {value.nbytes / 1024**2:.1f} MB')
        
        logging.info('\nFirst batch label info:')
        logging.info(f'  Labels shape: {batch_labels.shape}')
        logging.info(f'  Labels dtype: {batch_labels.dtype}')
        logging.info(f'  First sample - sum: {batch_labels[0].sum()}, active: {np.where(batch_labels[0] > 0)[0]}')
        
        active_classes = np.where(batch_labels[0] > 0)[0]
        logging.info(f'\nFirst sample has {len(active_classes)} active classes:')
        for class_idx in active_classes[:5]:
          logging.info(f'  - {index_to_name[class_idx]}')
      
      # Add batch to accumulator (handles batch dimension internally)
      accumulator.add_sample(essential, batch_labels)
      
      # Delete ALL batch data immediately after processing to free RAM
      del batch_labels
      del essential
      del inputs
      del batch  # Delete the entire batch dict including inputs
      
      # Force immediate garbage collection after deleting batch data
      import gc
      gc.collect()
      
      processed_count += current_batch_size
      batch_count += 1
      
      batch_time = time.time() - batch_start_time
      logging.info(f'  → Batch completed in {batch_time:.1f}s ({current_batch_size / batch_time:.2f} samples/sec)')
      
      # Clear JAX cache and run garbage collection MORE aggressively
      if batch_count % FLAGS.clear_cache_every == 0:
        jax.clear_caches()
        import gc
        gc.collect()
        # Force Python to release memory back to OS
        try:
          import ctypes
          libc = ctypes.CDLL("libc.so.6")
          libc.malloc_trim(0)
        except Exception:
          pass  # malloc_trim not available on all systems
      
      # Save checkpoint periodically
      if FLAGS.checkpoint_every > 0 and batch_count % FLAGS.checkpoint_every == 0 and batch_count > 0:
        save_checkpoint(accumulator, processed_count, FLAGS.output_dir)
        # Aggressively clean up memory after checkpoint
        import gc
        gc.collect()
        gc.collect()  # Run twice to clean up cyclic references
        # Force return memory to OS
        try:
          import ctypes
          libc = ctypes.CDLL("libc.so.6")
          libc.malloc_trim(0)
        except Exception:
          pass
    
    except Exception as e:
      logging.error(f'ERROR processing batch {batch_idx}: {e}')
      import traceback
      logging.error(traceback.format_exc())
      
      # Save emergency checkpoint on error
      if FLAGS.checkpoint_every > 0:
        logging.info('Saving emergency checkpoint before continuing...')
        save_checkpoint(accumulator, processed_count, FLAGS.output_dir, 'checkpoint_emergency.pkl')
      
      processed_count += current_batch_size
      continue
  return None
  # # Get stats WITHOUT loading all averages into memory
  # stats = accumulator.get_stats()
  
  # logging.info(f'\nAccumulation Statistics:')
  # logging.info(f'  Classes with samples: {stats["num_classes_with_samples"]}/{num_classes}')
  # logging.info(f'  Samples per class - min: {stats["min_samples"]}, max: {stats["max_samples"]}, mean: {stats["mean_samples"]:.1f}')
  
  # # Save class-averaged activations streaming to avoid OOM
  # # NOTE: We don't load all data into memory at once because ~88GB >> 62GB RAM
  # logging.info('\nSaving class-averaged activations (streaming to avoid OOM)...')
  # output_path = os.path.join(FLAGS.output_dir, 'class_averaged_activations.npz')
  
  # # Instead of using np.savez_compressed with a dict (which loads everything),
  # # we'll write incrementally using a context manager approach
  # # Unfortunately np.savez doesn't support streaming, so we'll use zarr or save to individual files
  
  # # OPTION 1: Save each class as separate file (best for memory)
  # # This is the most memory-efficient approach

  # averaged_dir = os.path.join(FLAGS.output_dir, 'averaged_activations')
  # os.makedirs(averaged_dir, exist_ok=True)
  
  # total_size = 0
  # count = 0
  
  # # Copy .npy files from accumulation and divide by counts on-the-fly
  # for class_idx in sorted(accumulator.counts.keys()):
  #   count_val = accumulator.counts[class_idx]
    
  #   for act_name in accumulator.activation_names:
  #     sum_path = accumulator._get_sum_path(class_idx, act_name)
  #     if os.path.exists(sum_path):
  #       # Load individual sum, divide by count, save to output
  #       act_sum = np.load(sum_path)
  #       avg_act = act_sum / count_val
        
  #       # Save averaged version
  #       avg_path = os.path.join(averaged_dir, f'class_{class_idx}_{act_name}.npy')
  #       np.save(avg_path, avg_act)
  #       total_size += avg_act.nbytes
  #       count += 1
        
  #       if count % 1000 == 0:
  #         logging.info(f'  Saved {count} averaged activations')
        
  #       # Clean up to free memory
  #       del act_sum, avg_act
  
  # logging.info(f'  Saved {count} averaged activations')
  # logging.info(f'  Total size: {total_size / (1024**3):.2f} GB')
  # logging.info(f'  Location: {averaged_dir}/')
  
  # # Also create a metadata file for easy loading
  # metadata_for_averages = {
  #     'class_names': np.array([index_to_name.get(i, '') for i in range(num_classes)], dtype=object),
  #     'class_mids': np.array([index_to_mid.get(i, '') for i in range(num_classes)], dtype=object),
  #     'samples_per_class': np.array([stats['samples_per_class'].get(i, 0) for i in range(num_classes)]),
  #     'num_classes': num_classes,
  #     'num_samples_processed': processed_count,
  #     'activation_names': accumulator.activation_names,
  #     'class_indices_with_samples': sorted(list(accumulator.counts.keys()))
  # }
  # metadata_path = os.path.join(averaged_dir, 'metadata.pkl')
  # with open(metadata_path, 'wb') as f:
  #   pickle.dump(metadata_for_averages, f)
  
  # logging.info(f'  Saved metadata to {metadata_path}')
  
  # # Keep temporary accumulation files (do NOT delete)
  # # accumulator.cleanup_accumulation_files()  # Disabled to preserve .accumulation files
  # logging.info('Keeping .accumulation directory for inspection/debugging')
  
  # # Clean up checkpoint files after successful completion
  # # if FLAGS.checkpoint_every > 0:
  # #   checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint.pkl')
  # #   emergency_checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint_emergency.pkl')
  # #   if os.path.exists(checkpoint_path):
  # #     os.remove(checkpoint_path)
  # #     logging.info('Removed checkpoint file (no longer needed)')
  # #   if os.path.exists(emergency_checkpoint_path):
  # #     os.remove(emergency_checkpoint_path)
  # #     logging.info('Removed emergency checkpoint file (no longer needed)')
  # # Checkpoint not removed to preserve resumption ability.

  # # Save detailed class statistics
  # class_stats = []
  # for class_idx in range(num_classes):
  #   count = stats['samples_per_class'].get(class_idx, 0)
  #   class_stats.append({
  #       'index': class_idx,
  #       'mid': index_to_mid.get(class_idx, ''),
  #       'display_name': index_to_name.get(class_idx, ''),
  #       'num_samples': count
  #   })
  
  # stats_df = pd.DataFrame(class_stats)
  # stats_path = os.path.join(FLAGS.output_dir, 'class_statistics.csv')
  # stats_df.to_csv(stats_path, index=False)
  
  # # Save metadata
  # metadata = {
  #     'checkpoint_dir': FLAGS.checkpoint_dir,
  #     'test_data_dir': FLAGS.test_data_dir,
  #     'num_samples_processed': processed_count,
  #     'num_classes': num_classes,
  #     'config': config.to_dict(),
  #     'statistics': stats
  # }
  # metadata_path = os.path.join(FLAGS.output_dir, 'metadata.pkl')
  # with open(metadata_path, 'wb') as f:
  #   pickle.dump(metadata, f)
  
  # logging.info('\n' + '='*80)
  # logging.info('Class-Averaged Extraction Complete!')
  # logging.info(f'Processed {processed_count} samples')
  # logging.info(f'Computed averages for {stats["num_classes_with_samples"]} classes')
  # logging.info(f'Total storage: {total_size / (1024**3):.2f} GB')
  # logging.info(f'Output saved to: {output_path}')
  # logging.info('='*80)
  
  # # Print usage instructions
  # logging.info('\nTo load class-averaged activations:')
  # logging.info('  import numpy as np')
  # logging.info(f'  data = np.load("{output_path}")')
  # logging.info('  # Get activation for class 137 (Music), layer 0, RGB:')
  # logging.info('  music_L0_rgb = data["class_137_encoder_block_L0_rgb_output"]')
  # logging.info('  # Get class names:')
  # logging.info('  class_names = data["class_names"]')
  # logging.info('  samples_per_class = data["samples_per_class"]')
  
  # # Show top 10 classes by sample count
  # logging.info('\nTop 10 classes by sample count:')
  # top_classes = sorted(stats['samples_per_class'].items(), key=lambda x: x[1], reverse=True)[:10]
  # for class_idx, count in top_classes:
  #   logging.info(f'  {index_to_name[class_idx]}: {count} samples')


if __name__ == '__main__':
  app.run(main)
