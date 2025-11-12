#!/usr/bin/env python3
"""Simple inference script to extract activations from MBT model.

This script provides a straightforward way to:
1. Load a trained MBT checkpoint
2. Run inference on test samples
3. Extract and save layer activations

Usage:
  python extract_activations.py \
    --checkpoint_path=mbt_base \
    --test_data_path=Audioset_test/data-00000-of-00001.tfrecord \
    --output_dir=activations_output

The script will save:
- activations_sample_XXXX.npz: Layer activations for each sample
- predictions.npz: Model predictions for all samples
- config.pkl: Configuration used
"""

import os
import pickle
from absl import app, flags, logging
import jax
import jax.numpy as jnp
import numpy as np

# Import scenic modules
from scenic.projects.mbt import model as mbt_model
from scenic.projects.mbt import train_utils as mbt_train_utils
from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset as av_dataset
from scenic.projects.mbt.configs.audioset import Inference_config
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', 'mbt_base',
                    'Path to model checkpoint directory')
flags.DEFINE_string('test_data_path', 'Audioset_test/data-00000-of-00001.tfrecord',
                    'Path to test TFRecord file')
flags.DEFINE_string('output_dir', 'activations_output',
                    'Directory to save activation outputs')
flags.DEFINE_integer('num_samples', 9,
                     'Number of test samples to process')


def load_test_data_properly(config):
  """Load test samples using the proper dataset infrastructure.
  
  Args:
    config: Configuration dict with dataset settings
    
  Returns:
    Dataset iterator
  """
  # Create dataset factory
  ds_factory = av_dataset.AVTFRecordDatasetFactory(
      base_dir=config.dataset_configs.base_dir,
      tables=config.dataset_configs.tables,
      num_classes=config.dataset_configs.num_classes,
      examples_per_subset=config.dataset_configs.examples_per_subset,
      subset='test',
      modalities=config.dataset_configs.modalities,
      prop_data=1.0,
  )
  
  # Load the test split using DMVR
  dataset, num_examples = av_dataset.load_split_from_dmvr(
      ds_factory=ds_factory,
      batch_size=1,  # Process one at a time
      subset='test',
      modalities=config.dataset_configs.modalities,
      num_frames=config.dataset_configs.num_frames,
      stride=config.dataset_configs.stride,
      num_spec_frames=config.dataset_configs.num_spec_frames,
      spec_stride=config.dataset_configs.spec_stride,
      num_test_clips=1,  # Single clip for analysis
      min_resize=config.dataset_configs.min_resize,
      crop_size=config.dataset_configs.crop_size,
      spec_shape=config.dataset_configs.spec_shape,
      dataset_spec_mean=config.dataset_configs.spec_mean,
      dataset_spec_stddev=config.dataset_configs.spec_stddev,
      spec_augment=False,  # No augmentation during inference
      spec_augment_params=None,
      one_hot_label=config.dataset_configs.one_hot_labels,
      zero_centering=config.dataset_configs.zero_centering,
      augmentation_params=None,  # No augmentation
  )
  
  logging.info(f'Loaded dataset with {num_examples} examples')
  return dataset, num_examples


def create_model_and_load_checkpoint(checkpoint_path):
  """Create model and load trained checkpoint.
  
  Args:
    checkpoint_path: Path to checkpoint directory
    
  Returns:
    Tuple of (model, train_state, config)
  """
  # Get the inference config
  config = Inference_config.get_config()
  
  # Build model
  logging.info('Building MBT model...')
  model = mbt_model.MBTMultiLabelClassificationModel(config, {
      'num_classes': config.dataset_configs.num_classes,
      'input_shape': {
          'rgb': (-1, config.dataset_configs.num_frames, 224, 224, 3),
          'spectrogram': (-1, config.dataset_configs.num_spec_frames, 128, 3)
      },
      'input_dtype': jnp.float32,
      'target_is_onehot': True
  })
  
  # Initialize model
  rng = jax.random.PRNGKey(0)
  rng, init_rng = jax.random.split(rng)
  
  input_spec = {
      'rgb': ((-1, config.dataset_configs.num_frames, 224, 224, 3), jnp.float32),
      'spectrogram': ((-1, config.dataset_configs.num_spec_frames, 128, 3), jnp.float32)
  }
  
  (params, model_state, num_trainable_params,
   gflops) = mbt_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)
  
  logging.info(f'Model: {num_trainable_params:,} parameters, {gflops} GFLOPs')
  
  # Create train state
  from scenic.train_lib import optimizers, lr_schedules
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  optimizer = optimizers.get_optimizer(optimizer_config, learning_rate_fn, params)
  opt_state = optimizer.init(params)
  
  train_state = train_utils.TrainState(
      global_step=0,
      tx=optimizer,
      params=params,
      opt_state=opt_state,
      model_state=model_state,
      rng=rng,
      metadata={})
  
  # Load checkpoint
  if os.path.exists(checkpoint_path):
    logging.info(f'Loading checkpoint from {checkpoint_path}')
    train_state, _ = train_utils.restore_checkpoint(checkpoint_path, train_state)
    logging.info('Checkpoint loaded successfully')
  else:
    logging.warning(f'Checkpoint not found at {checkpoint_path}, using random init')
  
  return model, train_state, config


def extract_activations_with_intermediate_capture(model, train_state, inputs):
  """Extract activations using Flax's capture_intermediates.
  
  Args:
    model: MBT model instance
    train_state: Training state with parameters
    inputs: Input dict with 'rgb' and 'spectrogram' keys
    
  Returns:
    Dictionary with activations and predictions
  """
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }
  
  # Run with intermediate capture
  # This captures outputs from all intermediate layers
  output, collected = model.flax_model.apply(
      variables,
      inputs,
      train=False,
      mutable=False,
      capture_intermediates=lambda module, method_name: method_name == '__call__'
  )
  
  # Parse collected intermediates
  activations = {}
  if 'intermediates' in collected:
    for path, value in collected['intermediates'].items():
      # Convert to numpy and store
      layer_name = '/'.join(str(p) for p in path)
      if hasattr(value['__call__'], 'shape'):
        activations[layer_name] = np.array(value['__call__'])
  
  return {
      'logits': np.array(output),
      'activations': activations
  }


def main(argv):
  del argv  # Unused
  
  logging.info('='*80)
  logging.info('MBT Activation Extraction')
  logging.info('='*80)
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Create config from Inference_config
  logging.info('\n[1/3] Loading configuration...')
  config = Inference_config.get_config()
  
  # Load model
  logging.info('\n[2/3] Loading model and checkpoint...')
  model, train_state, _ = create_model_and_load_checkpoint(FLAGS.checkpoint_path)
  
  # Load test data
  logging.info('\n[3/3] Loading test data...')
  dataset, num_examples = load_test_data_properly(config)
  logging.info(f'Loaded dataset with {num_examples} examples')
  
  # Process each sample
  logging.info(f'\nExtracting activations from {num_examples} samples...')
  all_results = []
  
  sample_idx = 0
  for batch in dataset.take(num_examples):
    logging.info(f'  Processing sample {sample_idx+1}/{num_examples}...')
    
    # Extract inputs from batch
    inputs = batch['inputs']
    
    # Extract activations
    result = extract_activations_with_intermediate_capture(
        model, train_state, inputs
    )
    result['sample_idx'] = sample_idx
    all_results.append(result)
    
    # Save individual sample
    output_path = os.path.join(FLAGS.output_dir, f'activations_sample_{sample_idx:04d}.npz')
    np.savez_compressed(
        output_path,
        logits=result['logits'],
        **{f'activation_{k}': v for k, v in result['activations'].items()}
    )
    logging.info(f'    Saved to {output_path}')
    logging.info(f'    Captured {len(result["activations"])} layers')
    
    sample_idx += 1
  
  # Save combined results
  logging.info('\nSaving combined results...')
  combined_path = os.path.join(FLAGS.output_dir, 'all_predictions.npz')
  np.savez_compressed(
      combined_path,
      logits=np.array([r['logits'] for r in all_results])
  )
  logging.info(f'Saved predictions to {combined_path}')
  
  # Save config
  config_path = os.path.join(FLAGS.output_dir, 'config.pkl')
  with open(config_path, 'wb') as f:
    pickle.dump(config.to_dict(), f)
  
  logging.info('\n' + '='*80)
  logging.info('Extraction complete!')
  logging.info(f'Outputs saved to: {FLAGS.output_dir}')
  logging.info('='*80)


if __name__ == '__main__':
  app.run(main)
