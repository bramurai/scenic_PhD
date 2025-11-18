#!/usr/bin/env python3
"""Extract activations and attention weights from trained MBT model.

This script:
1. Loads a trained MBT checkpoint (not ViT pretrained, but your trained MBT model)
2. Runs forward pass on test samples
3. Extracts all layer activations and attention weights
4. Saves outputs for PCA, attention flow analysis, etc.

Usage:
  python extract_mbt_activations.py \
    --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
    --checkpoint_dir=mbt_base \
    --test_data_dir=/media/labuta/7f1ad7d2-a1d3-4a1f-ae81-7cb5dd2661a3/VGG_Preprocessed/test_tfrecords_local \
    --output_dir=activation_analysis \
    --num_samples=100
"""

import os
import pickle
from typing import Dict, Any, Tuple
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from flax.training import checkpoints
import tensorflow as tf

# Scenic imports
from scenic.projects.mbt import model as mbt_model
from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('config', None, 'Path to config file (e.g., vggsound_base.py)')
flags.DEFINE_string('checkpoint_dir', None, 
                    'Directory containing checkpoint files (e.g., CheckPoints/mbt_run1). '
                    'This should be the folder containing the checkpoint files, not the CheckPoints folder itself.')
flags.DEFINE_string('test_data_dir', None, 'Directory with test TFRecords')
flags.DEFINE_string('output_dir', 'activation_analysis', 'Output directory')
flags.DEFINE_integer('num_samples', 100, 'Number of samples to process')
flags.DEFINE_integer('checkpoint_step', None, 
                     'Specific checkpoint step to load (e.g., 1000, 5000). '
                     'If None, loads the latest checkpoint.')

flags.mark_flag_as_required('config')
flags.mark_flag_as_required('checkpoint_dir')
flags.mark_flag_as_required('test_data_dir')


def load_config(config_path: str) -> ml_collections.ConfigDict:
  """Load config from Python file."""
  import importlib.util
  spec = importlib.util.spec_from_file_location("config", config_path)
  config_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config_module)
  config = config_module.get_config()
  
  # Override dataset path for test data
  config.dataset_configs.base_dir = FLAGS.test_data_dir
  config.dataset_configs.tables = {
      'test': [os.path.join(FLAGS.test_data_dir, 'tar*', '*.tfrecord')],
  }
  
  return config


def create_test_dataset(config: ml_collections.ConfigDict):
  """Create test dataset iterator."""
  logging.info('Creating test dataset...')
  
  # Find all test TFRecords
  import glob
  import functools
  tfrecord_pattern = os.path.join(FLAGS.test_data_dir, '**', '*.tfrecord')
  tfrecord_files = glob.glob(tfrecord_pattern, recursive=True)
  
  if not tfrecord_files:
    raise ValueError(f'No TFRecords found in {FLAGS.test_data_dir}')
  
  logging.info(f'Found {len(tfrecord_files)} TFRecord files')
  
  # Convert to relative paths from test_data_dir
  tfrecord_files_relative = [os.path.relpath(f, FLAGS.test_data_dir) for f in tfrecord_files]
  
  # Create dataset factory using functools.partial (not an instance)
  ds_factory = functools.partial(
      audiovisual_tfrecord_dataset.AVTFRecordDatasetFactory,
      base_dir=FLAGS.test_data_dir,
      tables={'test': tfrecord_files_relative},
      num_classes=config.dataset_configs.num_classes,
      examples_per_subset={'test': FLAGS.num_samples},
      num_groups=1,
      group_index=0
  )
  
  # Load dataset
  dataset, num_examples = audiovisual_tfrecord_dataset.load_split_from_dmvr(
      ds_factory=ds_factory,
      batch_size=1,  # Process one at a time
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
  
  # Apply post-processing transformations (map_keys, etc.)
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
  
  # Load checkpoint first
  # Check if checkpoint_dir is actually a file (for single checkpoint files)
  if os.path.isfile(checkpoint_dir):
    logging.info(f'Loading checkpoint file: {checkpoint_dir}')
    checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
  elif FLAGS.checkpoint_step is not None:
    checkpoint_path = checkpoints.restore_checkpoint(
        checkpoint_dir, 
        None, 
        step=FLAGS.checkpoint_step
    )
  else:
    # Try to find checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if not f.startswith('.') and os.path.isfile(os.path.join(checkpoint_dir, f))]
    
    if len(checkpoint_files) == 1:
      # Single checkpoint file in directory
      single_ckpt = os.path.join(checkpoint_dir, checkpoint_files[0])
      logging.info(f'Found single checkpoint file: {single_ckpt}')
      checkpoint_path = checkpoints.restore_checkpoint(single_ckpt, None)
    else:
      # Standard Scenic checkpoint directory with checkpoint_* files
      checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
  
  if checkpoint_path is None:
    raise ValueError(f'No checkpoint found in {checkpoint_dir}')
  
  # Extract params from checkpoint
  if 'params' in checkpoint_path:
    params = checkpoint_path['params']
  elif 'optimizer' in checkpoint_path and 'target' in checkpoint_path['optimizer']:
    params = checkpoint_path['optimizer']['target']
  else:
    params = checkpoint_path
  
  # Create model (no initialization needed, we have params from checkpoint)
  model_cls = mbt_model.MBTMultilabelClassificationModel
  
  # Spectrogram shape: num_spec_frames chunks are concatenated along time dimension
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
  
  model_state = {}  # No batch_stats needed for inference
  rng = jax.random.PRNGKey(0)
  
  logging.info('Checkpoint loaded successfully')
  return model_instance, params, model_state, rng


def compute_attention_weights(activations: Dict) -> Dict:
  """Compute attention weights from query and key activations.
  
  For each attention layer, compute:
    attention_weights = softmax(Q @ K^T / sqrt(d_k))
    
  where Q and K are the query and key tensors.
  """
  attention_weights = {}
  
  # Find all attention layers by looking for query/key pairs
  attention_layers = {}
  for key in activations.keys():
    if 'MultiHeadDotProductAttention' in key:
      # Extract base name (remove _query, _key, _value, etc.)
      base_name = key.rsplit('_', 2)[0] if key.endswith(('_query___call___0', '_key___call___0', '_value___call___0')) else None
      if base_name and base_name not in attention_layers:
        attention_layers[base_name] = {}
      
      if base_name:
        if key.endswith('_query___call___0'):
          attention_layers[base_name]['query'] = activations[key]
        elif key.endswith('_key___call___0'):
          attention_layers[base_name]['key'] = activations[key]
  
  # Compute attention weights for each layer
  for layer_name, qk_dict in attention_layers.items():
    if 'query' in qk_dict and 'key' in qk_dict:
      Q = qk_dict['query']  # Shape: (batch, seq_len, num_heads, head_dim)
      K = qk_dict['key']    # Shape: (batch, seq_len, num_heads, head_dim)
      
      # Transpose for attention computation
      # Q: (batch, num_heads, seq_len, head_dim)
      # K: (batch, num_heads, head_dim, seq_len)
      Q_t = np.transpose(Q, (0, 2, 1, 3))
      K_t = np.transpose(K, (0, 2, 3, 1))
      
      # Compute attention scores: Q @ K^T / sqrt(d_k)
      head_dim = Q.shape[-1]
      scores = np.matmul(Q_t, K_t) / np.sqrt(head_dim)
      
      # Apply softmax
      # Subtract max for numerical stability
      scores_max = np.max(scores, axis=-1, keepdims=True)
      exp_scores = np.exp(scores - scores_max)
      attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
      
      # Save with a clean name
      clean_name = layer_name.replace('activation_', '')
      attention_weights[f'attention_weights_{clean_name}'] = attention
  
  return attention_weights


def extract_with_intermediates(model_instance, params, model_state, inputs):
  """Extract activations using intermediate capture.
  
  Returns dict with:
    - logits: Model predictions
    - intermediates: All intermediate layer outputs
    - attention_weights: Attention weights from each layer (if available)
  """
  variables = {'params': params}
  if model_state:
    variables['batch_stats'] = model_state
  
  # Apply with intermediate capture
  output, state = model_instance.flax_model.apply(
      variables,
      inputs,
      train=False,
      mutable=['intermediates'],
      capture_intermediates=True
  )
  
  # Extract intermediates
  activations = {}
  attention_weights = {}
  
  def extract_from_dict(d, prefix='', depth=0):
    """Recursively extract arrays from nested FrozenDicts."""
    if depth == 0:
      logging.info("Exploring intermediates structure...")
    
    # Check if it's dict-like (has items()) or tuple-like
    has_items = hasattr(d, 'items')
    is_tuple = isinstance(d, tuple)
    
    if has_items or is_tuple:
      items = d.items() if has_items else enumerate(d)
      for key, value in items:
        new_prefix = f"{prefix}/{key}" if prefix else str(key)
        
        # Log what we're seeing
        if depth < 3:  # Only log first few levels
          logging.info(f"{'  ' * depth}{new_prefix}: {type(value).__name__}")
        
        if hasattr(value, 'shape'):
          # It's an array
          activations[new_prefix] = np.array(value)
          logging.info(f"✓ Captured activation: {new_prefix}, shape={value.shape}")
        else:
          # Recurse into nested structure
          extract_from_dict(value, new_prefix, depth + 1)
  
  if 'intermediates' in state:
    intermediates = state['intermediates']
    logging.info(f"Found intermediates: type={type(intermediates)}, len={len(intermediates) if hasattr(intermediates, '__len__') else 'N/A'}")
    logging.info(f"Intermediates repr (first 500 chars): {str(intermediates)[:500]}")
    
    extract_from_dict(intermediates)
    logging.info(f"Total activations captured: {len(activations)}")
    
    # Compute attention weights from query and key activations
    logging.info("Computing attention weights from Q and K...")
    attention_weights = compute_attention_weights(activations)
    logging.info(f"Computed {len(attention_weights)} attention weight matrices")
  else:
    logging.warning("No intermediates found in state!")
  
  return {
      'logits': np.array(output),
      'activations': activations,
      'attention_weights': attention_weights
  }


def filter_essential_activations(activations: Dict, attention_weights: Dict) -> Tuple[Dict, Dict, Dict]:
  """Filter to keep only essential activations.
  
  Keeps:
    1. Final output of each encoder block (24 total: 12 RGB + 12 Audio)
    2. Bottleneck tokens (from layers 8-11)
    3. Full attention weight matrices (24 total)
  
  Returns:
    (encoder_outputs, bottleneck_tokens, attention_matrices)
  """
  encoder_outputs = {}
  bottleneck_tokens = {}
  attention_matrices = {}
  
  logging.info("\nFiltering essential activations...")
  
  # Pattern for final encoder block outputs
  # Keys look like: "Transformer/encoderblock_0/MlpBlock_0/__call__/0"
  # Or: "Transformer/encoderblock_0_spectrogram/MlpBlock_0/__call__/0"
  for key, value in activations.items():
    if 'encoderblock_' in key and 'MlpBlock_0/__call__/0' in key and 'Transformer' in key:
      # Extract layer number - key format: "Transformer/encoderblock_10_spectrogram/MlpBlock..."
      # or "Transformer/encoderblock_10/MlpBlock..."
      parts = key.split('/')
      for part in parts:
        if part.startswith('encoderblock_'):
          # Extract layer number and check for spectrogram
          if '_spectrogram' in part:
            layer_num = part.replace('encoderblock_', '').replace('_spectrogram', '')
            name = f'encoder_block_L{layer_num}_audio_output'
          else:
            layer_num = part.replace('encoderblock_', '')
            name = f'encoder_block_L{layer_num}_rgb_output'
          encoder_outputs[name] = value
          logging.info(f"  Found {name}: shape {value.shape}")
          break
  
  # Extract bottleneck tokens from fused sequences (layers 8-11)
  # The bottleneck tokens are the last 5 tokens in the sequence for layers >= 8
  for key, value in activations.items():
    if 'encoderblock_' in key and 'MlpBlock_0/__call__/0' in key and 'Transformer' in key:
      parts = key.split('/')
      for part in parts:
        if part.startswith('encoderblock_'):
          # Extract layer number
          if '_spectrogram' in part:
            layer_num_str = part.replace('encoderblock_', '').replace('_spectrogram', '')
            try:
              layer_num = int(layer_num_str)
              if layer_num >= 8:
                bottleneck_tokens[f'bottleneck_L{layer_num}_audio'] = value[:, -5:, :]
                logging.info(f"  Extracted bottleneck_L{layer_num}_audio: shape {value[:, -5:, :].shape}")
            except ValueError:
              pass
          else:
            layer_num_str = part.replace('encoderblock_', '')
            try:
              layer_num = int(layer_num_str)
              if layer_num >= 8:
                bottleneck_tokens[f'bottleneck_L{layer_num}_rgb'] = value[:, -5:, :]
                logging.info(f"  Extracted bottleneck_L{layer_num}_rgb: shape {value[:, -5:, :].shape}")
            except ValueError:
              pass
          break
  
  # Compute attention weights for each encoder block
  logging.info("\nComputing attention weights for encoder blocks...")
  attention_layers = {}
  
  # Find query and key for each attention layer
  # Pattern: "Transformer/encoderblock_10/MultiHeadDotProductAttention_0/query/__call__/0"
  for key in activations.keys():
    if 'MultiHeadDotProductAttention' in key and 'encoderblock_' in key and 'Transformer' in key:
      # Extract base layer identifier from path
      parts = key.split('/')
      for part in parts:
        if part.startswith('encoderblock_'):
          base_name = part  # e.g., "encoderblock_10" or "encoderblock_10_spectrogram"
          if base_name not in attention_layers:
            attention_layers[base_name] = {}
          
          if '/query/__call__/0' in key:
            attention_layers[base_name]['query'] = activations[key]
          elif '/key/__call__/0' in key:
            attention_layers[base_name]['key'] = activations[key]
          break
  
  # Compute attention weights from Q and K
  for layer_name, qk_dict in attention_layers.items():
    if 'query' in qk_dict and 'key' in qk_dict:
      Q = qk_dict['query']  # Shape: (batch, seq_len, num_heads, head_dim)
      K = qk_dict['key']    # Shape: (batch, seq_len, num_heads, head_dim)
      
      # Transpose for attention computation
      Q_t = np.transpose(Q, (0, 2, 1, 3))  # (batch, num_heads, seq_len, head_dim)
      K_t = np.transpose(K, (0, 2, 3, 1))  # (batch, num_heads, head_dim, seq_len)
      
      # Compute attention: softmax(Q @ K^T / sqrt(d_k))
      head_dim = Q.shape[-1]
      scores = np.matmul(Q_t, K_t) / np.sqrt(head_dim)
      
      # Apply softmax
      scores_max = np.max(scores, axis=-1, keepdims=True)
      exp_scores = np.exp(scores - scores_max)
      attention = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
      
      # Create clean name
      if 'spectrogram' in layer_name:
        # Extract layer number
        layer_num = layer_name.split('_')[1]
        name = f'attention_weights_L{layer_num}_audio'
      else:
        layer_num = layer_name.split('_')[1]
        name = f'attention_weights_L{layer_num}_rgb'
      
      attention_matrices[name] = attention
      logging.info(f"  Computed {name}: shape {attention.shape}")
  
  logging.info("\nFiltered activations:")
  logging.info(f"  Encoder outputs: {len(encoder_outputs)}")
  logging.info(f"  Bottleneck tokens: {len(bottleneck_tokens)}")
  logging.info(f"  Attention matrices: {len(attention_matrices)}")
  
  return encoder_outputs, bottleneck_tokens, attention_matrices


def save_sample_data(sample_data: Dict, sample_idx: int, output_dir: str):
  """Save activations for one sample."""
  output_path = os.path.join(output_dir, f'sample_{sample_idx:05d}.npz')
  
  # Filter to essential activations only
  encoder_outputs, bottleneck_tokens, attention_matrices = filter_essential_activations(
      sample_data['activations'],
      sample_data.get('attention_weights', {})
  )
  
  # Prepare data for saving
  save_dict = {
      'logits': sample_data['logits'],
      'sample_idx': sample_idx
  }
  
  # Add encoder block outputs
  for name, activation in encoder_outputs.items():
    save_dict[name] = activation
  
  # Add bottleneck tokens
  for name, tokens in bottleneck_tokens.items():
    save_dict[name] = tokens
  
  # Add attention weights
  for name, weights in attention_matrices.items():
    save_dict[name] = weights
  
  # Log what we're saving
  total_size = sum(arr.nbytes for arr in save_dict.values() if hasattr(arr, 'nbytes'))
  logging.info(f"Saving sample {sample_idx}:")
  logging.info(f"  Total items: {len(save_dict)}")
  logging.info(f"  Total size: {total_size / (1024**2):.1f} MB")
  
  np.savez_compressed(output_path, **save_dict)
  return output_path


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('MBT Activation Extraction')
  logging.info('='*80)
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Load config
  logging.info('\n[1/4] Loading configuration...')
  config = load_config(FLAGS.config)
  
  # Save config
  config_path = os.path.join(FLAGS.output_dir, 'config.pkl')
  with open(config_path, 'wb') as f:
    pickle.dump(config.to_dict(), f)
  
  # Load checkpoint
  logging.info('\n[2/4] Loading checkpoint...')
  model_instance, params, model_state, rng = load_checkpoint(config, FLAGS.checkpoint_dir)
  
  # Create dataset
  logging.info('\n[3/4] Loading test data...')
  dataset, num_examples = create_test_dataset(config)
  num_to_process = min(FLAGS.num_samples, num_examples)
  
  # Process samples
  logging.info(f'\n[4/4] Extracting activations from {num_to_process} samples...')
  
  all_logits = []
  activation_summary = {}
  
  # dataset is now an iterator, not a TF dataset
  for sample_idx, batch in enumerate(dataset):
    if sample_idx >= num_to_process:
      break
      
    if sample_idx % 10 == 0:
      logging.info(f'  Processing sample {sample_idx+1}/{num_to_process}...')
    
    # Extract activations
    inputs = batch['inputs']
    result = extract_with_intermediates(model_instance, params, model_state, inputs)
    
    # Save individual sample
    output_path = save_sample_data(result, sample_idx, FLAGS.output_dir)
    
    # Collect summary stats
    all_logits.append(result['logits'])
    
    if sample_idx == 0:
      # Log layer information
      logging.info(f'\n  Extracted {len(result["activations"])} activation layers:')
      for name, activation in result['activations'].items():
        logging.info(f'    {name}: {activation.shape}')
      
      if result['attention_weights']:
        logging.info(f'\n  Extracted {len(result["attention_weights"])} attention weight matrices:')
        for name, weights in result['attention_weights'].items():
          logging.info(f'    {name}: {weights.shape}')
  
  # Save summary
  logging.info('\nSaving summary...')
  summary_path = os.path.join(FLAGS.output_dir, 'summary.npz')
  np.savez_compressed(
      summary_path,
      logits=np.array(all_logits),
      num_samples=num_to_process
  )
  
  # Save metadata
  metadata = {
      'checkpoint_dir': FLAGS.checkpoint_dir,
      'test_data_dir': FLAGS.test_data_dir,
      'num_samples': num_to_process,
      'config': config.to_dict()
  }
  metadata_path = os.path.join(FLAGS.output_dir, 'metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
  
  logging.info('\n' + '='*80)
  logging.info('Extraction Complete!')
  logging.info(f'Processed {num_to_process} samples')
  logging.info(f'Outputs saved to: {FLAGS.output_dir}')
  logging.info('='*80)
  
  # Print usage instructions
  logging.info('\nTo load activations in Python:')
  logging.info('  import numpy as np')
  logging.info(f'  data = np.load("{FLAGS.output_dir}/sample_00000.npz")')
  logging.info('  logits = data["logits"]')
  logging.info('  activation = data["activation_<layer_name>"]')


if __name__ == '__main__':
  app.run(main)
