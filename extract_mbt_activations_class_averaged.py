#!/usr/bin/env python3
"""Extract class-averaged activations and logits from trained MBT model.

This script:
1. Loads a trained MBT checkpoint
2. Runs forward pass on all test samples
3. Optionally accumulates activations per class (handles multi-label properly)
4. Optionally extracts and saves final logits
5. Optionally computes mean Average Precision (mAP)
6. Saves only class-averaged data (~26 GB instead of ~1.9 TB for activations)

For multi-label samples:
  - Each sample contributes to the average of ALL its classes
  - Example: Sample with labels [Music, Speech] contributes to both Music and Speech averages

Storage comparison:
  - Per-sample storage: 3,853 samples × 0.5 GB = 1.9 TB
  - Class-averaged storage: 527 classes × ~50 MB = 26 GB (100x smaller!)

Usage:
  # Extract everything (activations, logits, and compute mAP):
  python extract_mbt_activations_class_averaged.py \
    --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
    --checkpoint_dir=CheckPoints/MBT_AV \
    --test_data_dir=Datasets/audioset_evel_configCorrect \
    --output_dir=audioset_analysis_12-9-2025 \
    --audioset_labels_csv=Video_csvs/audioset_labels.csv\
    --save_logits \
    --compute_map \
    --batch_size=1 \
    --num_samples=500

  # Extract only logits and compute mAP (no activations):
  python extract_mbt_activations_class_averaged.py \
    --config=... --checkpoint_dir=... --test_data_dir=... \
    --output_dir=audioset_logits_only \
    --audioset_labels_csv=... \
    --nosave_activations \
    --save_logits \
    --compute_map
"""

import os
import pickle
import time
import functools
import glob
import importlib.util
from typing import Dict, Any
import math
from collections import defaultdict
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import ml_collections
from flax.training import checkpoints
from flax import jax_utils
import tensorflow as tf
import subprocess

# Scenic imports
from scenic.projects.mbt import model as mbt_model
from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset


class GPUMemoryTracker:
  """Track GPU memory usage across batches."""
  
  def __init__(self):
    self.batch_memory_used = []  # Memory used per batch
    self.peak_memory = 0
    self.num_gpus = self._get_num_gpus()
    self.oom_batch_size = None  # Track which batch size caused OOM
    self.num_oom_errors = 0  # Count OOM errors
  
  def _get_num_gpus(self):
    """Get number of available GPUs."""
    try:
      result = subprocess.run(
          ['nvidia-smi', '--list-gpus'],
          capture_output=True,
          text=True,
          check=True
      )
      return len(result.stdout.strip().split('\n'))
    except:
      return 0
  
  def get_memory_used_mb(self):
    """Get current GPU memory usage in MB."""
    try:
      result = subprocess.run(
          ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
          capture_output=True,
          text=True,
          check=True
      )
      values = [int(x.strip()) for x in result.stdout.strip().split('\n')]
      return sum(values)  # Total across all GPUs
    except:
      return 0
  
  def get_memory_total_mb(self):
    """Get total GPU memory in MB."""
    try:
      result = subprocess.run(
          ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
          capture_output=True,
          text=True,
          check=True
      )
      values = [int(x.strip()) for x in result.stdout.strip().split('\n')]
      return sum(values)  # Total across all GPUs
    except:
      return 0
  
  def log_batch_memory(self, batch_num, num_samples):
    """Log memory usage for a batch."""
    mem_used = self.get_memory_used_mb()
    if mem_used > self.peak_memory:
      self.peak_memory = mem_used
    self.batch_memory_used.append(mem_used)
    
    mem_per_sample = mem_used / num_samples if num_samples > 0 else 0
    return mem_used, mem_per_sample
  
  def get_stats(self):
    """Get memory statistics."""
    if not self.batch_memory_used:
      return {}
    
    return {
        'peak_memory_mb': self.peak_memory,
        'avg_memory_mb': np.mean(self.batch_memory_used),
        'min_memory_mb': min(self.batch_memory_used),
        'max_memory_mb': max(self.batch_memory_used),
        'total_gpu_memory_mb': self.get_memory_total_mb(),
        'num_gpus': self.num_gpus,
    }
  
  def record_oom_error(self, batch_size):
    """Record that an OOM error occurred at this batch size."""
    self.num_oom_errors += 1
    if self.oom_batch_size is None:
      self.oom_batch_size = batch_size
      logging.warning(f'OOM error recorded at batch size {batch_size}')
  
  def estimate_max_batch_size(self, memory_per_sample_mb, current_batch_size, safety_margin=0.15):
    """Estimate maximum batch size based on current usage and OOM history."""
    total_memory = self.get_memory_total_mb()
    
    # If we had OOM errors at a certain batch size, be conservative
    if self.oom_batch_size is not None:
      # Recommend reducing batch size if OOM occurred
      recommended = max(1, self.oom_batch_size // 2)
      return recommended, f"OOM occurred at batch size {self.oom_batch_size}, reducing recommendation to {recommended}"
    
    # Normal estimation based on memory per sample
    available = total_memory * (1 - safety_margin)
    used_per_crop = memory_per_sample_mb
    
    # Account for 4 crops per sample (from num_test_clips)
    max_batch_size = int(available / (used_per_crop * 4))
    
    # Never recommend increasing beyond current if we're already using significant memory
    memory_usage_pct = (self.peak_memory / total_memory) * 100 if total_memory > 0 else 0
    if memory_usage_pct > 50:
      # High memory usage - be conservative
      return max(1, current_batch_size), f"Memory usage is {memory_usage_pct:.1f}%, keeping batch size at current level"
    
    return max(1, max_batch_size), None


FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'Path to config file')
flags.DEFINE_string('checkpoint_dir', None, 'Directory containing checkpoint files')
flags.DEFINE_string('test_data_dir', None, 'Directory with test TFRecords')
flags.DEFINE_string('output_dir', 'audioset_class_averaged', 'Output directory')
flags.DEFINE_string('audioset_labels_csv', None, 'Path to audioset_labels.csv for class names')
flags.DEFINE_integer('num_samples', None, 'Number of samples to process (None = all)')
flags.DEFINE_integer('batch_size', 4, 'Batch size for Pass 1 (device-parallel logits, should equal num_devices=4)')
flags.DEFINE_integer('pass2_batch_size', 2, 'Batch size for Pass 2 (activation extraction, higher = faster)')
flags.DEFINE_bool('device_parallel_logits', True, 'Run logits/mAP pass with pmap across devices (trainer parity)')
flags.DEFINE_bool('two_pass', True, 'Two-pass: first logits-only with pmap, then sequential activations to avoid OOM')
flags.DEFINE_bool('average_attention_heads', True, 'Average attention over heads to reduce size')
flags.DEFINE_integer('clear_cache_every', 1, 'Clear JAX cache every N samples')
flags.DEFINE_bool('save_attention', False, 'Save attention weights (increases storage significantly)')
flags.DEFINE_bool('save_activations', True, 'Save encoder block activations (disable to save storage)')
flags.DEFINE_bool('save_logits', True, 'Save final model logits')
flags.DEFINE_bool('compute_map', True, 'Compute and report mean Average Precision (mAP)')
flags.DEFINE_integer('checkpoint_every', 1, 'Save intermediate checkpoint every N batches (0 = disable)')
flags.DEFINE_bool('resume_from_checkpoint', True, 'Resume from checkpoint if it exists')

flags.mark_flag_as_required('config')
flags.mark_flag_as_required('checkpoint_dir')
flags.mark_flag_as_required('test_data_dir')
flags.mark_flag_as_required('audioset_labels_csv')


def load_config(config_path: str) -> ml_collections.ConfigDict:
  """Load config from Python file."""
  spec = importlib.util.spec_from_file_location("config", config_path)
  config_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config_module)
  config = config_module.get_config()
  
  config.dataset_configs.base_dir = FLAGS.test_data_dir
  config.dataset_configs.tables = {
      'test': [os.path.join(FLAGS.test_data_dir, 'tar*', '*.tfrecord')],
  }
  
  return config


def read_labels_from_tfrecord(tfrecord_path: str, return_ids: bool = False) -> list:
  """Read ALL multi-hot labels from a TFRecord file.
  
  Args:
    tfrecord_path: Path to TFRecord file
    return_ids: If True, also return video IDs for debugging
    
  Returns:
    List of multi-hot label vectors (or tuples of (label, video_id) if return_ids=True)
  """
  labels = []
  
  # Read ALL records from this file (not just the first one)
  for raw_record in tf.data.TFRecordDataset([tfrecord_path]):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    # Read multi-hot label from context
    if 'clip/label/multi_hot' in example.context.feature:
      label_floats = example.context.feature['clip/label/multi_hot'].float_list.value
      label_array = np.array(label_floats, dtype=np.float32)
      
      if return_ids:
        # Try to get video ID for debugging
        video_id = ''
        if 'clip/label/text' in example.context.feature:
          video_id = example.context.feature['clip/label/text'].bytes_list.value[0].decode('utf-8')
        elif 'example/id' in example.context.feature:
          video_id = example.context.feature['example/id'].bytes_list.value[0].decode('utf-8')
        labels.append((label_array, video_id))
      else:
        labels.append(label_array)
    else:
      raise ValueError(f'No clip/label/multi_hot found in {tfrecord_path}')
  
  return labels


def create_test_dataset(config: ml_collections.ConfigDict, batch_size: int = None, num_test_clips: int = None):
  """Create test dataset iterator with labels read separately.
  
  Args:
    config: Configuration dict
    batch_size: Override batch size (defaults to FLAGS.batch_size)
    num_test_clips: Override num_test_clips (defaults to config value)
  """
  logging.info('Creating test dataset...')
  
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
  
  # Use provided or default values
  actual_batch_size = batch_size if batch_size is not None else FLAGS.batch_size
  actual_num_test_clips = num_test_clips if num_test_clips is not None else config.dataset_configs.get('num_test_clips', 1)
  
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
      batch_size=actual_batch_size,
      subset='test',
      modalities=('spectrogram', 'rgb'),
      num_frames=config.dataset_configs.num_frames,
      stride=config.dataset_configs.stride,
      num_spec_frames=config.dataset_configs.num_spec_frames,
      spec_stride=config.dataset_configs.spec_stride,
      num_test_clips=actual_num_test_clips,
      min_resize=config.dataset_configs.min_resize,
      crop_size=config.dataset_configs.crop_size,
      spec_shape=config.dataset_configs.spec_shape,
      dataset_spec_mean=config.dataset_configs.get('spec_mean', 0.0),
      dataset_spec_stddev=config.dataset_configs.get('spec_stddev', 1.0),
      spec_augment=False,  # Always False for inference
      spec_augment_params=None,  # Always None for inference
      one_hot_label=config.dataset_configs.get('one_hot_labels', True),
      zero_centering=config.dataset_configs.get('zero_centering', True),
      augmentation_params=None,  # Always None for inference
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


def resolve_modality_keys(batch: Dict[str, Any]):
  """Resolve modality keys present in the batch for rgb and spectrogram.

  Returns a dict mapping canonical names to actual keys in the batch.
  """
  keys = set(batch.keys())
  rgb_candidates = ['rgb', 'image']
  spec_candidates = ['spectrogram', 'spec', 'mel', 'audio']
  rgb_key = next((k for k in rgb_candidates if k in keys), None)
  spec_key = next((k for k in spec_candidates if k in keys), None)
  return {'rgb': rgb_key, 'spectrogram': spec_key}


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


def pmapped_test_step_factory(flax_model, num_classes: int):
  """Create a pmapped test step that mirrors trainer semantics (logits-only).

  Each device receives `clips_per_device` crops for a single example per step,
  sums logits over those crops, and we iterate chunks until all crops are
  covered; finally average over total crops per example. Intermediates are NOT
  captured to keep memory safe.
  """

  def per_device_step(variables, inputs_chunk):
    # inputs_chunk shapes: {'rgb': [clips_per_device, ...], 'spectrogram': [...]} per device
    # When mutable=False, apply() returns just the output (not a tuple)
    output = flax_model.apply(variables, inputs_chunk, train=False, mutable=False)
    # Sum logits across the `clips_per_device` axis
    return jnp.sum(output, axis=0)

  return jax.pmap(per_device_step, axis_name='devices')


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


def process_sample_with_multicrop(params, model_state, inputs, flax_model, num_test_clips, n_clips=2):
  """Process a sample with multiple crops - matches trainer.py test_step pattern.
  
  This is a simplified version that follows the original test_step exactly,
  but also captures intermediate activations for MLP neuron extraction.
  
  Args:
    params: Model parameters
    model_state: Model state (batch_stats)
    inputs: Dict with modality inputs - shape [num_crops, ...]
    flax_model: Flax model instance
    num_test_clips: Total number of crops (e.g., 4)
    n_clips: Number of crops to process at once (e.g., 2)
  
  Returns:
    Dict with 'logits' and 'activations' averaged across all crops
  """
  variables = {'params': params}
  if model_state:
    variables['batch_stats'] = model_state
  
  # Initialize accumulator for logits
  all_logits = jnp.zeros(527)  # AudioSet has 527 classes
  all_activations = {}
  
  # Filter function: ONLY capture MLP outputs, NOT attention matrices (which are huge!)
  def capture_mlp_only(module, method_name):
    """Only capture MLP block outputs, skip attention to save massive memory."""
    # Handle modules with no name
    if module.name is None:
      return False
    
    # Skip attention completely - these create 903MB matrices!
    if 'MultiHeadDotProductAttention' in module.name or 'Attention' in module.name:
      return False
    
    # Capture MLP block outputs (these contain the neuron activations we need)
    if 'MlpBlock' in module.name:
      return True
    
    # Skip everything else (embeddings, LayerNorm, Dropout, query/key/value, etc.)
    return False
  
  # Process crops in chunks - exact pattern from trainer.py test_step
  for idx in range(0, num_test_clips, n_clips):
    # Extract chunk of crops for this iteration
    current_input = {}
    for modality in inputs:
      current_input[modality] = inputs[modality][idx:idx + n_clips]
    
    # Forward pass with filtered intermediate capture - ONLY MLP, not attention
    output, state = flax_model.apply(
        variables, current_input, train=False,
        mutable=['intermediates'], capture_intermediates=capture_mlp_only
    )
    
    # Accumulate logits (sum, then we'll average at end)
    logits_sum = jnp.sum(output, axis=0)
    all_logits = all_logits + logits_sum
    
    # Extract and accumulate activations from intermediates
    # MOVE TO CPU IMMEDIATELY to free GPU memory
    if 'intermediates' in state:
      
      def extract_from_dict(d, prefix=''):
        """Recursively extract arrays from nested FrozenDicts and move to CPU."""
        if hasattr(d, 'items'):
          for key, value in d.items():
            new_prefix = f'{prefix}/{key}' if prefix else key
            
            if isinstance(value, dict) or hasattr(value, 'items'):
              extract_from_dict(value, new_prefix)
            # Handle tuples (Flax stores method outputs as tuples)
            elif isinstance(value, tuple):
              # If tuple contains a single array, extract it
              if len(value) == 1 and isinstance(value[0], (jnp.ndarray, np.ndarray)):
                array = value[0]
                chunk_sum = jnp.sum(array, axis=0)
                chunk_sum_cpu = np.array(chunk_sum)  # Copy to CPU/numpy
                if new_prefix in all_activations:
                  all_activations[new_prefix] = all_activations[new_prefix] + chunk_sum_cpu
                else:
                  all_activations[new_prefix] = chunk_sum_cpu
              # If multiple elements, try each
              else:
                for i, elem in enumerate(value):
                  if isinstance(elem, (jnp.ndarray, np.ndarray)):
                    elem_prefix = f'{new_prefix}_{i}'
                    chunk_sum = jnp.sum(elem, axis=0)
                    chunk_sum_cpu = np.array(chunk_sum)
                    if elem_prefix in all_activations:
                      all_activations[elem_prefix] = all_activations[elem_prefix] + chunk_sum_cpu
                    else:
                      all_activations[elem_prefix] = chunk_sum_cpu
            elif isinstance(value, (jnp.ndarray, np.ndarray)):
              # Sum across crops in this chunk AND move to CPU immediately
              chunk_sum = jnp.sum(value, axis=0)
              chunk_sum_cpu = np.array(chunk_sum)  # Copy to CPU/numpy
              if new_prefix in all_activations:
                all_activations[new_prefix] = all_activations[new_prefix] + chunk_sum_cpu
              else:
                all_activations[new_prefix] = chunk_sum_cpu
      
      extract_from_dict(state['intermediates'])
      # Delete GPU state immediately after extracting
      del state
  
  # Average logits and activations across all crops
  averaged_logits = all_logits / num_test_clips
  averaged_activations = {k: v / num_test_clips for k, v in all_activations.items()}
  
  return {
      'logits': averaged_logits,
      'activations': averaged_activations
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
    - encoder_block_L0_rgb_output (if save_activations=True)
    - encoder_block_L0_audio_output (if save_activations=True)
    - attention_weights_L0_rgb (if save_attention=True)
  """
  essential = {}
  
  # Extract encoder block outputs (conditional on save_activations flag)
  if FLAGS.save_activations:
    for key, value in activations.items():
      if 'encoderblock_' in key and 'MlpBlock_0/__call__' in key and 'Transformer' in key:
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
  
  # Sync to disk
  os.sync()
  logging.info(f'  Saved checkpoint to {checkpoint_path}')


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

            # Load existing sum, add to it, save back
            existing_sum = np.load(sum_path)
            existing_sum[:] += act_value
            np.save(sum_path, existing_sum)
          else:
            # First time seeing this class - save initial sum

            np.save(sum_path, np.array(act_value, copy=True, dtype=np.float32))
    
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

              # Load existing sum, add to it, save back
              existing_sum = np.load(sum_path)
              existing_sum[:] += sample_activation
              np.save(sum_path, existing_sum)
            else:

              # First time seeing this class - save initial sum
              np.save(sum_path, np.array(sample_activation, copy=True, dtype=np.float32))
  
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


class LogitsAccumulator:
  """Accumulates logits and labels for mAP computation and class-averaged logits."""
  
  def __init__(self, num_classes: int, output_dir: str):
    self.num_classes = num_classes
    self.output_dir = output_dir
    
    # For mAP computation - accumulate all predictions and labels
    self.all_logits = []
    self.all_labels = []
    
    # For class-averaged logits
    self.logit_sums = defaultdict(lambda: np.zeros(num_classes, dtype=np.float32))
    self.logit_counts = defaultdict(int)
  
  def add_sample(self, logits: np.ndarray, labels: np.ndarray):
    """Add sample logits and labels.
    
    Args:
      logits: Logit predictions - shape: (batch, num_classes) or (num_classes,)
      labels: Multi-hot label vector(s) - shape: (batch, num_classes) or (num_classes,)
    """
    # Handle both single sample and batched inputs
    if labels.ndim == 1:
      # Single sample
      self.all_logits.append(logits)
      self.all_labels.append(labels)
      
      active_classes = np.where(labels > 0)[0]
      for class_idx in active_classes:
        class_idx = int(class_idx)
        self.logit_sums[class_idx] += logits
        self.logit_counts[class_idx] += 1
    else:
      # Batched samples
      batch_size = labels.shape[0]
      self.all_logits.extend([logits[i] for i in range(batch_size)])
      self.all_labels.extend([labels[i] for i in range(batch_size)])
      
      for batch_idx in range(batch_size):
        label = labels[batch_idx]
        logit = logits[batch_idx]
        active_classes = np.where(label > 0)[0]
        
        for class_idx in active_classes:
          class_idx = int(class_idx)
          self.logit_sums[class_idx] += logit
          self.logit_counts[class_idx] += 1
  
  def compute_map(self):
    """Compute mean Average Precision using sklearn."""
    from sklearn.metrics import average_precision_score
    
    if not self.all_logits:
      logging.warning('No logits accumulated for mAP computation')
      return None
    
    # Convert lists to arrays
    logits_array = np.array(self.all_logits, dtype=np.float32)
    labels_array = np.array(self.all_labels, dtype=np.float32)
    
    # Use raw logits directly (matching original evaluation_lib.compute_mean_average_precision)
    # sklearn's average_precision_score handles raw scores - sigmoid not needed
    
    # Compute per-class AP - store with class indices
    aps_per_class = {}  # Maps class_idx -> AP score
    for class_idx in range(self.num_classes):
      y_true = labels_array[:, class_idx]
      y_score = logits_array[:, class_idx]  # Use raw logits
      
      # Only compute AP if there are positive samples
      if y_true.sum() > 0:
        ap = average_precision_score(y_true, y_score)
        aps_per_class[class_idx] = ap
    
    # Compute mAP (average over classes that have samples)
    map_score = np.mean(list(aps_per_class.values())) if aps_per_class else 0.0
    return map_score, aps_per_class
  
  def get_averaged_logits(self):
    """Get class-averaged logits."""
    averaged = {}
    for class_idx, logit_sum in self.logit_sums.items():
      count = self.logit_counts[class_idx]
      if count > 0:
        averaged[class_idx] = logit_sum / count
    return averaged
  
  def save_logits(self, output_dir: str):
    """Save all logits and labels for later analysis."""
    logits_path = os.path.join(output_dir, 'all_logits.npz')
    
    logits_array = np.array(self.all_logits, dtype=np.float32)
    labels_array = np.array(self.all_labels, dtype=np.float32)
    
    np.savez_compressed(logits_path, 
                       logits=logits_array,
                       labels=labels_array)
    
    logging.info(f'Saved all logits to {logits_path}')
    logging.info(f'  Shape: {logits_array.shape}')
    logging.info(f'  Size: {logits_array.nbytes / 1024**2:.1f} MB')
    
    # Also save class-averaged logits
    averaged_logits = self.get_averaged_logits()
    averaged_logits_path = os.path.join(output_dir, 'class_averaged_logits.npz')
    
    # Convert dict to arrays for saving
    class_indices = sorted(averaged_logits.keys())
    averaged_array = np.array([averaged_logits[idx] for idx in class_indices], dtype=np.float32)
    
    np.savez_compressed(averaged_logits_path,
                       logits=averaged_array,
                       class_indices=np.array(class_indices, dtype=np.int32))
    
    logging.info(f'Saved class-averaged logits to {averaged_logits_path}')
    logging.info(f'  Shape: {averaged_array.shape}')


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('MBT Class-Averaged Activation Extraction')
  logging.info('='*80)
  
  # Memory management handled by HPC system
  
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
  
  # Determine device count and evaluation strategy
  num_devices = jax.local_device_count()
  logging.info(f"\nDevices detected: {num_devices}")
  
  # Pass 1: Always use batch_size=4 for device-parallel pmap (4 GPUs)
  pass1_batch_size = 4
  pass1_num_test_clips = config.dataset_configs.get('num_test_clips', 4)
  
  # Pass 2: Use configurable batch size, NO multicrop (single crop for speed)
  pass2_batch_size = FLAGS.pass2_batch_size
  pass2_num_test_clips = 1  # No multicrop in Pass 2 = 4x faster!
  
  logging.info(f'Pass 1 config: batch_size={pass1_batch_size}, num_crops={pass1_num_test_clips} (device-parallel with pmap)')
  logging.info(f'Pass 2 config: batch_size={pass2_batch_size}, num_crops={pass2_num_test_clips} (sequential, no multicrop)')
  
  multicrop_clips_per_device = config.dataset_configs.get('multicrop_clips_per_device', 2)
  
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
  
  # Initialize accumulators
  accumulator = ClassAccumulator(num_classes, FLAGS.output_dir)
  
  # Initialize GPU memory tracker
  gpu_tracker = GPUMemoryTracker()
  logging.info(f'GPU Memory Tracking initialized ({gpu_tracker.num_gpus} GPUs, {gpu_tracker.get_memory_total_mb():.0f} MB total)')
  
  # Initialize logits accumulator if needed
  logits_accumulator = None
  if FLAGS.save_logits or FLAGS.compute_map:
    logits_accumulator = LogitsAccumulator(num_classes, FLAGS.output_dir)
  
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
  
  # Pass 1: Device-parallel logits/mAP (batch_size=4, multicrop=4)
  if FLAGS.two_pass:
    logging.info('\n[Pass 1] Device-parallel logits/mAP evaluation (no activations)')
    
    # Create Pass 1 dataset with batch_size=4 and multicrop=4
    dataset_pass1, _, _ = create_test_dataset(config, batch_size=pass1_batch_size, num_test_clips=pass1_num_test_clips)
    # Replicate variables
    variables = {'params': params}
    if model_state:
      variables['batch_stats'] = model_state
    replicated_vars = jax_utils.replicate(variables)

    per_device_step = pmapped_test_step_factory(model_instance.flax_model, num_classes)

    # Iterate dataset once for logits
    examples_seen = 0
    for b_idx, batch in enumerate(dataset_pass1):
      if examples_seen >= num_to_process:
        break
      inputs_dict = batch.get('inputs', batch)
      key_map = resolve_modality_keys(inputs_dict)
      if key_map['rgb'] is None or key_map['spectrogram'] is None:
        logging.error(f"Batch modalities missing; keys available: {list(inputs_dict.keys())}. Expected one of rgb/image and spectrogram/spec.")
        raise KeyError('Required modalities not found in batch')
      # Expect batch_size == num_devices, one example per device
      # batch['rgb']: [batch_size, num_test_clips, ...]
      # We process crops in chunks of `multicrop_clips_per_device` per step
      clips_per_device = multicrop_clips_per_device
      total_chunks = math.ceil(pass1_num_test_clips / clips_per_device)
      logits_sum_per_example = np.zeros((pass1_batch_size, num_classes), dtype=np.float32)

      for chunk_idx in range(total_chunks):
        start = chunk_idx * clips_per_device
        end = min(start + clips_per_device, pass1_num_test_clips)
        # Slice per device chunk: [devices, clips_per_device, ...]
        inputs_chunk = {}
        # Use resolved keys; normalize shapes if crops are flattened
        def ensure_batch_clip_shape(x):
          if x.shape[0] == pass1_batch_size and x.ndim >= 2 and x.shape[1] == pass1_num_test_clips:
            return x
          if x.shape[0] == pass1_batch_size * pass1_num_test_clips:
            new_shape = (pass1_batch_size, pass1_num_test_clips) + x.shape[1:]
            return x.reshape(new_shape)
          raise ValueError(f"Unexpected input shape {x.shape}; cannot infer [batch, num_test_clips, ...] layout")

        x_rgb_full = inputs_dict[key_map['rgb']]
        x_spec_full = inputs_dict[key_map['spectrogram']]
        x_rgb_bc = ensure_batch_clip_shape(x_rgb_full)
        x_spec_bc = ensure_batch_clip_shape(x_spec_full)
        inputs_chunk['rgb'] = x_rgb_bc[:, start:end]
        inputs_chunk['spectrogram'] = x_spec_bc[:, start:end]

        # Run pmapped step: returns [devices, num_classes]
        logits_sum_devices = per_device_step(replicated_vars, inputs_chunk)
        logits_sum_devices_np = np.array(logits_sum_devices)
        logits_sum_per_example += logits_sum_devices_np

      averaged_logits_examples = logits_sum_per_example / float(pass1_num_test_clips)

      # Labels: Use dataset labels (now correctly loaded from clip/label/multi_hot)
      # Dataset labels are guaranteed to match the data since they come from the same pipeline
      dataset_labels = batch.get('label', batch.get('inputs', {}).get('label', None))
      
      if dataset_labels is not None and dataset_labels.sum() > 0:
        # Use dataset labels (correct approach)
        labels_np = np.array(dataset_labels)
      else:
        # Fallback: Read from TFRecords sequentially (may be misaligned, but better than nothing)
        batch_labels_list = []
        for i in range(pass1_batch_size):
          try:
            label = next(label_iterator)
            batch_labels_list.append(label)
          except StopIteration:
            logging.warning(f'Label iterator exhausted at example {examples_seen}')
            break
        labels_np = np.array(batch_labels_list) if batch_labels_list else None
      
      if labels_np is not None:
        # DEBUG: Log first batch to verify label alignment
        if b_idx == 0:
          logging.info(f'\n[Pass 1 DEBUG] First batch label verification:')
          logging.info(f'  Batch index: {b_idx}')
          logging.info(f'  Sample indices: {examples_seen} to {examples_seen + len(labels_np) - 1}')
          
          # CRITICAL CHECK: Show EVERYTHING in the batch to diagnose missing labels
          logging.info(f'\n  === RAW BATCH INSPECTION ===')
          logging.info(f'  Batch ALL keys: {list(batch.keys())}')
          for key in batch.keys():
            val = batch[key]
            if hasattr(val, 'shape'):
              logging.info(f'    {key}: shape={val.shape}, dtype={val.dtype}')
            elif isinstance(val, dict):
              logging.info(f'    {key}: dict with keys={list(val.keys())}')
              for subkey in val.keys():
                subval = val[subkey]
                if hasattr(subval, 'shape'):
                  logging.info(f'      {key}[{subkey}]: shape={subval.shape}, dtype={subval.dtype}')
            else:
              logging.info(f'    {key}: type={type(val).__name__}')
          
          # Check for dataset labels (for debugging/verification only)
          if dataset_labels is not None:
            logging.info(f'\n  ⚠️  Dataset contains labels with shape {dataset_labels.shape}')
            logging.info(f'  Dataset label dtype: {dataset_labels.dtype}')
            logging.info(f'  Dataset label stats: min={dataset_labels.min()}, max={dataset_labels.max()}, mean={dataset_labels.mean()}')
            logging.info(f'  Dataset label sum (total): {dataset_labels.sum()}')
            logging.info(f'  ✅ Using dataset labels for all batches (guaranteed to match data)')
          else:
            logging.info(f'\n  ℹ️  Dataset does NOT contain labels (falling back to TFRecord streaming)')
              
          for i, label in enumerate(labels_np):
            active_classes = np.where(label > 0)[0]
            class_names = [index_to_name.get(c, f'Unknown_{c}') for c in active_classes[:3]]
            logging.info(f'  Sample {examples_seen + i}: {len(active_classes)} classes - {class_names}')
            # Also log top 3 logit predictions for comparison
            top_logit_indices = np.argsort(averaged_logits_examples[i])[-3:][::-1]
            top_logit_classes = [index_to_name.get(c, f'Unknown_{c}') for c in top_logit_indices]
            top_logit_scores = averaged_logits_examples[i][top_logit_indices]
            logging.info(f'  Sample {examples_seen + i}: Top 3 predictions - {list(zip(top_logit_classes, top_logit_scores))}')
            
            # Check if ANY prediction is in ground truth
            overlap = set(top_logit_indices[:10]) & set(active_classes)
            logging.info(f'  Sample {examples_seen + i}: Top-10 overlap with GT = {len(overlap)}/{len(active_classes)} classes')
        
        if logits_accumulator is not None:
          logits_accumulator.add_sample(averaged_logits_examples, labels_np)

      examples_seen += pass1_batch_size
      logging.info(f'[Pass 1] Processed {examples_seen}/{num_to_process} examples')

    logging.info('[Pass 1] Device-parallel logits evaluation complete')
    
    # Report Pass 1 mAP
    if FLAGS.compute_map and logits_accumulator is not None:
      try:
        map_score, aps_per_class = logits_accumulator.compute_map()
        logging.info(f'[Pass 1] mAP: {map_score:.4f}')
        logging.info(f'[Pass 1] Classes with predictions: {len(aps_per_class)}/{num_classes}')
      except Exception as e:
        logging.warning(f'[Pass 1] Failed to compute mAP: {e}')

    # Recreate dataset iterator for pass 2 with batch_size=1 to avoid OOM
    logging.info(f'\n[Pass 2] Sequential activation extraction (batch_size={pass2_batch_size}, no multicrop for speed)')
    
    # CRITICAL: Reset label iterator for Pass 2 (Pass 1 already consumed it)
    label_iterator = create_label_iterator(tfrecord_files)
    logging.info('  Reset label iterator for Pass 2 (starting from beginning)')
    
    # Create Pass 2 dataset with configurable batch size and NO multicrop (num_test_clips=1)
    dataset, _, _ = create_test_dataset(config, batch_size=pass2_batch_size, num_test_clips=pass2_num_test_clips)

  # Pass 2: Sequential activation extraction with configurable batch size
  for batch_idx, batch in enumerate(dataset):
    if processed_count >= num_to_process:
      break
    
    batch_start_time = time.time()
    
    # Resolve modality keys for sequential processing as well
    inputs_dict_seq = batch.get('inputs', batch)
    key_map_seq = resolve_modality_keys(inputs_dict_seq)
    if key_map_seq['rgb'] is None or key_map_seq['spectrogram'] is None:
      logging.error(f"Batch modalities missing; keys available: {list(inputs_dict_seq.keys())}. Expected one of rgb/image and spectrogram/spec.")
      raise KeyError('Required modalities not found in batch')

    # Get batch size from actual data (Pass 2: batch_size * 1 crop)
    actual_batch_size = inputs_dict_seq[key_map_seq['rgb']].shape[0] // pass2_num_test_clips
    
    # Get labels from dataset (guaranteed to match data)
    batch_labels = batch.get('label', batch.get('inputs', {}).get('label', None))
    
    # Log progress
    elapsed = time.time() - start_time
    speed = processed_count / elapsed if elapsed > 0 else 0
    remaining = num_to_process - processed_count
    eta_seconds = remaining / speed if speed > 0 else 0
    eta_str = f'{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m' if speed > 0 else 'calculating...'
    
    logging.info(
        f'Batch {batch_idx}: Processing samples {processed_count}-{processed_count + actual_batch_size}/{num_to_process} '\
        f'({pass2_num_test_clips} crop) | Speed: {speed:.2f} samples/sec | '\
        f'GPU Memory: {gpu_tracker.get_memory_used_mb():.0f} MB | ETA: {eta_str}'\
    )
    
    # Process each sample in batch (single crop in Pass 2)
    for sample_idx in range(actual_batch_size):
      if processed_count >= num_to_process:
        break
      
      # Get label for this sample from dataset batch
      if batch_labels is not None:
        label = batch_labels[sample_idx]
      else:
        # Fallback to TFRecord streaming if dataset doesn't have labels
        try:
          label = next(label_iterator)
        except StopIteration:
          logging.warning(f'Label iterator exhausted at sample {processed_count}')
          break
      
      # Extract this sample's crop (only 1 crop in Pass 2)
      sample_inputs = {}
      for modality_key in inputs_dict_seq:
        if modality_key in ['label', 'batch_mask']:
          continue
        # Shape: [batch_size * 1, ...] -> extract single crop
        start_idx = sample_idx * pass2_num_test_clips
        end_idx = (sample_idx + 1) * pass2_num_test_clips
        sample_inputs[modality_key] = inputs_dict_seq[modality_key][start_idx:end_idx]
      
      # Process sample (single crop in Pass 2, so just 1 forward pass)
      result = process_sample_with_multicrop(
          params, model_state, sample_inputs, 
          model_instance.flax_model, pass2_num_test_clips, multicrop_clips_per_device
      )
      
      # Filter activations to essential (MLP outputs only)
      essential = filter_essential_activations(result['activations'])
      
      # Log first sample info
      if processed_count == 0:
        logging.info('\nFirst batch activations:')
        for name, value in essential.items():
          logging.info(f'  {name}: shape {value.shape}, size {value.nbytes / 1024**2:.1f} MB')
        
        logging.info('\nFirst batch label info:')
        logging.info(f'  Labels shape: {label.shape}')
        logging.info(f'  Labels dtype: {label.dtype}')
        logging.info(f'  First sample - sum: {label.sum()}, active: {np.where(label > 0)[0]}')
        
        active_classes = np.where(label > 0)[0]
        logging.info(f'\nFirst sample has {len(active_classes)} active classes:')
        for class_idx in active_classes[:5]:
          logging.info(f'  - {index_to_name[class_idx]}')
      
      # Add to accumulator (handles multi-label correctly)
      if FLAGS.save_activations and essential:
        accumulator.add_sample(essential, label)
        del essential  # Free CPU memory immediately
      
      # Clean up
      del result, sample_inputs, label
      
      processed_count += 1
    
    # Batch completed
    batch_duration = time.time() - batch_start_time
    logging.info(f'  → Batch completed in {batch_duration:.1f}s ({actual_batch_size / batch_duration:.2f} samples/sec)')
    
    # Save checkpoint if requested
    if FLAGS.checkpoint_every > 0 and batch_idx % FLAGS.checkpoint_every == 0:
      save_checkpoint(accumulator, processed_count, FLAGS.output_dir)
    
    batch_count += 1

  # Get stats WITHOUT loading all averages into memory
  stats = accumulator.get_stats()
  
  logging.info('\nAccumulation Statistics:')
  logging.info(f'  Classes with samples: {stats["num_classes_with_samples"]}/{num_classes}')
  logging.info(f'  Samples per class - min: {stats["min_samples"]}, max: {stats["max_samples"]}, mean: {stats["mean_samples"]:.1f}')
  
  # Compute mAP if requested
  map_score = None  # Initialize for later reporting
  if FLAGS.compute_map and logits_accumulator is not None:
    logging.info('\nComputing mean Average Precision (mAP)...')
    try:
      map_score, aps_per_class = logits_accumulator.compute_map()
      logging.info(f'  mAP: {map_score:.4f}')
      logging.info(f'  Number of classes with samples: {len(aps_per_class)}/{num_classes}')
      
      # Show top and bottom 5 classes by AP
      if aps_per_class:
        # Now aps_per_class is a dict mapping class_idx -> AP
        ap_with_names = [(index_to_name[class_idx], ap) for class_idx, ap in aps_per_class.items()]
        ap_with_names.sort(key=lambda x: x[1], reverse=True)
        
        logging.info('\n  Top 5 classes by AP:')
        for name, ap in ap_with_names[:5]:
          logging.info(f'    {name}: {ap:.4f}')
        
        logging.info('\n  Bottom 5 classes by AP:')
        for name, ap in ap_with_names[-5:]:
          logging.info(f'    {name}: {ap:.4f}')
    except Exception as e:
      logging.error(f'Failed to compute mAP: {e}')
      import traceback
      logging.error(traceback.format_exc())
  
  # Save logits if requested
  if FLAGS.save_logits and logits_accumulator is not None:
    logging.info('\nSaving logits...')
    try:
      logits_accumulator.save_logits(FLAGS.output_dir)
    except Exception as e:
      logging.error(f'Failed to save logits: {e}')
      import traceback
      logging.error(traceback.format_exc())
  
  # Save class-averaged activations streaming to avoid OOM (only if save_activations=True)
  # NOTE: We don't load all data into memory at once because ~88GB >> 62GB RAM
  total_size = 0  # Initialize for later reporting
  if FLAGS.save_activations and accumulator.activation_names:

    
    # Instead of using np.savez_compressed with a dict (which loads everything),
    # we'll write incrementally using a context manager approach
    # Unfortunately np.savez doesn't support streaming, so we'll use zarr or save to individual files
    
    # OPTION 1: Save each class as separate file (best for memory)
    # This is the most memory-efficient approach

    averaged_dir = os.path.join(FLAGS.output_dir, 'averaged_activations')
    os.makedirs(averaged_dir, exist_ok=True)
    
    total_size = 0
    count = 0
    
    # Copy .npy files from accumulation and divide by counts on-the-fly
    for class_idx in sorted(accumulator.counts.keys()):
      count_val = accumulator.counts[class_idx]
      
      for act_name in accumulator.activation_names:
        sum_path = accumulator._get_sum_path(class_idx, act_name)
        if os.path.exists(sum_path):
          # Load individual sum, divide by count, save to output
          act_sum = np.load(sum_path)
          avg_act = act_sum / count_val
          
          # Save averaged version
          avg_path = os.path.join(averaged_dir, f'class_{class_idx}_{act_name}.npy')
          np.save(avg_path, avg_act)
          total_size += avg_act.nbytes
          count += 1
          
          if count % 1000 == 0:
            logging.info(f'  Saved {count} averaged activations')
          
          # Clean up to free memory
          del act_sum, avg_act
    
    logging.info(f'  Saved {count} averaged activations')
    logging.info(f'  Total size: {total_size / (1024**3):.2f} GB')
    logging.info(f'  Location: {averaged_dir}/')
    
    # Also create a metadata file for easy loading
    metadata_for_averages = {
        'class_names': np.array([index_to_name.get(i, '') for i in range(num_classes)], dtype=object),
        'class_mids': np.array([index_to_mid.get(i, '') for i in range(num_classes)], dtype=object),
        'samples_per_class': np.array([stats['samples_per_class'].get(i, 0) for i in range(num_classes)]),
        'num_classes': num_classes,
        'num_samples_processed': processed_count,
        'activation_names': accumulator.activation_names,
        'class_indices_with_samples': sorted(list(accumulator.counts.keys()))
    }
    metadata_path = os.path.join(averaged_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
      pickle.dump(metadata_for_averages, f)
    
    logging.info(f'  Saved metadata to {metadata_path}')
  elif not FLAGS.save_activations:
    logging.info('\nSkipping activation saving (save_activations=False)')
  
  # Keep temporary accumulation files (do NOT delete)
  # accumulator.cleanup_accumulation_files()  # Disabled to preserve .accumulation files
  logging.info('Keeping .accumulation directory for inspection/debugging')
  
  # Clean up checkpoint files after successful completion
  # if FLAGS.checkpoint_every > 0:
  #   checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint.pkl')
  #   emergency_checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint_emergency.pkl')
  #   if os.path.exists(checkpoint_path):
  #     os.remove(checkpoint_path)
  #     logging.info('Removed checkpoint file (no longer needed)')
  #   if os.path.exists(emergency_checkpoint_path):
  #     os.remove(emergency_checkpoint_path)
  #     logging.info('Removed emergency checkpoint file (no longer needed)')
  # Checkpoint not removed to preserve resumption ability.

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
  
  # Report what was extracted
  if FLAGS.save_activations and accumulator.activation_names:
    logging.info(f'Saved activations: {len(accumulator.activation_names)} types per class')
    logging.info(f'Total activation storage: {total_size / (1024**3):.2f} GB')
  
  if FLAGS.save_logits and logits_accumulator:
    logging.info('Saved logits: all_logits.npz and class_averaged_logits.npz')
  
  if FLAGS.compute_map and map_score is not None:
    logging.info(f'Model mAP: {map_score:.4f}')
  
  # Print GPU memory statistics
  gpu_stats = gpu_tracker.get_stats()
  if gpu_stats:
    logging.info('\n' + '='*80)
    logging.info('GPU MEMORY STATISTICS:')
    logging.info(f'  Peak GPU Memory: {gpu_stats["peak_memory_mb"]:.0f} MB')
    logging.info(f'  Average GPU Memory: {gpu_stats["avg_memory_mb"]:.0f} MB')
    logging.info(f'  Total GPU Memory Available: {gpu_stats["total_gpu_memory_mb"]:.0f} MB')
    logging.info(f'  Number of GPUs: {gpu_stats["num_gpus"]}')
    
    if gpu_tracker.num_oom_errors > 0:
      logging.warning(f'\n  ⚠️  OOM ERRORS DETECTED: {gpu_tracker.num_oom_errors} OOM failure(s)')
      logging.warning(f'  Failed batch size: {gpu_tracker.oom_batch_size}')
      logging.warning(f'  RECOMMENDATION: Reduce BATCH_SIZE to 1 and retry, or split dataset into smaller parts')
    
    # Estimate safe batch size for next run
    if processed_count > 0 and gpu_stats['peak_memory_mb'] > 0:
      mem_per_sample = gpu_stats['peak_memory_mb'] / processed_count
      estimated_batch_size, note = gpu_tracker.estimate_max_batch_size(mem_per_sample, FLAGS.batch_size, safety_margin=0.15)
      logging.info(f'\n  Memory per sample: {mem_per_sample:.1f} MB')
      logging.info(f'  Estimated safe batch size for next run: {estimated_batch_size}')
      logging.info(f'  Current batch size: {FLAGS.batch_size}')
      if note:
        logging.warning(f'  → {note}')
      elif estimated_batch_size > FLAGS.batch_size:
        logging.info(f'  → You can increase BATCH_SIZE to {estimated_batch_size} to improve speed')
      elif estimated_batch_size < FLAGS.batch_size:
        logging.warning(f'  → BATCH_SIZE {FLAGS.batch_size} is too high! Reduce to {estimated_batch_size}')
  
  logging.info('='*80)
  
  # Print usage instructions
  if FLAGS.save_activations:
    logging.info('\nTo load class-averaged activations:')
    logging.info('  import numpy as np')
    averaged_dir = os.path.join(FLAGS.output_dir, 'averaged_activations')
    logging.info(f'  # Load activation for class 137 (Music), layer 0, RGB:')
    logging.info(f'  music_L0_rgb = np.load("{averaged_dir}/class_137_encoder_block_L0_rgb_output.npy")')
  
  if FLAGS.save_logits:
    logging.info('\nTo load logits:')
    logging.info('  import numpy as np')
    logging.info(f'  data = np.load("{FLAGS.output_dir}/all_logits.npz")')
    logging.info('  logits = data["logits"]  # shape: (num_samples, num_classes)')
    logging.info('  labels = data["labels"]  # shape: (num_samples, num_classes)')
  
  # Show top 10 classes by sample count
  logging.info('\nTop 10 classes by sample count:')
  top_classes = sorted(stats['samples_per_class'].items(), key=lambda x: x[1], reverse=True)[:10]
  for class_idx, count in top_classes:
    logging.info(f'  {index_to_name[class_idx]}: {count} samples')


if __name__ == '__main__':
  app.run(main)
