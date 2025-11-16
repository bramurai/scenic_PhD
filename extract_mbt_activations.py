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
from typing import Dict, Any
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
  
  logging.info(f'Dataset created with {num_examples} examples')
  return dataset, num_examples


def load_checkpoint(config: ml_collections.ConfigDict, checkpoint_dir: str):
  """Load trained MBT checkpoint."""
  logging.info(f'Loading checkpoint from {checkpoint_dir}...')
  
  # Create model
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
  
  # Initialize model
  rng = jax.random.PRNGKey(0)
  init_rng, rng = jax.random.split(rng)
  
  # Dummy input for initialization
  # Spectrogram shape is (batch, num_spec_frames * spec_shape[0], spec_shape[1], 3)
  # The chunks are concatenated along the time dimension
  spec_time_dim = config.dataset_configs.num_spec_frames * config.dataset_configs.spec_shape[0]
  dummy_input = {
      'rgb': jnp.zeros((1, config.dataset_configs.num_frames, 224, 224, 3)),
      'spectrogram': jnp.zeros((1, spec_time_dim, config.dataset_configs.spec_shape[1], 3))
  }
  
  # Initialize
  variables = model_instance.flax_model.init(
      init_rng,
      dummy_input,
      train=False
  )
  
  # Load checkpoint
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
  
  model_state = variables.get('batch_stats', {})
  
  logging.info('Checkpoint loaded successfully')
  return model_instance, params, model_state, rng


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
  
  if 'intermediates' in state:
    for key, value in state['intermediates'].items():
      layer_name = str(key)
      
      # Store activation
      if hasattr(value, 'shape'):
        activations[layer_name] = np.array(value)
      
      # Check if this is an attention layer (contains attention weights)
      if 'attention' in layer_name.lower() or 'attn' in layer_name.lower():
        if isinstance(value, dict) and 'attention_weights' in value:
          attention_weights[layer_name] = np.array(value['attention_weights'])
  
  return {
      'logits': np.array(output),
      'activations': activations,
      'attention_weights': attention_weights
  }


def save_sample_data(sample_data: Dict, sample_idx: int, output_dir: str):
  """Save activations for one sample."""
  output_path = os.path.join(output_dir, f'sample_{sample_idx:05d}.npz')
  
  # Prepare data for saving
  save_dict = {
      'logits': sample_data['logits'],
      'sample_idx': sample_idx
  }
  
  # Add activations
  for name, activation in sample_data['activations'].items():
    safe_name = name.replace('/', '_').replace('.', '_')
    save_dict[f'activation_{safe_name}'] = activation
  
  # Add attention weights
  for name, weights in sample_data['attention_weights'].items():
    safe_name = name.replace('/', '_').replace('.', '_')
    save_dict[f'attention_{safe_name}'] = weights
  
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
  
  for sample_idx, batch in enumerate(dataset.take(num_to_process)):
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
