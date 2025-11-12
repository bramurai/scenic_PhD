#!/usr/bin/env python3
"""Inference script for extracting neural activations and attention weights.

Usage:
  python run_inference_analysis.py \
    --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
    --workdir=inference_outputs/

This script will:
1. Load the trained MBT model
2. Process all 9 test samples
3. Extract layer-wise activations
4. Extract attention weights
5. Save all data for analysis
"""

import os
import pickle
from absl import app, flags, logging
import jax
import jax.numpy as jnp
from flax import jax_utils
import numpy as np
from scenic import app as scenic_app
from scenic.projects.mbt import model as mbt_model
from scenic.train_lib import train_utils

FLAGS = flags.FLAGS


def extract_activations_from_model(
    model,
    train_state,
    batch,
    save_path
):
  """Extract and save activations from a single batch.
  
  Args:
    model: MBT model instance
    train_state: Training state with parameters
    batch: Input batch
    save_path: Where to save activations
  """
  variables = {
      'params': jax_utils.unreplicate(train_state.params),
      **jax_utils.unreplicate(train_state.model_state)
  }
  
  # Run forward pass
  # Note: We'll need to modify the model to return intermediate activations
  logits = model.flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      mutable=False,
      debug=False
  )
  
  # For now, save logits and inputs
  # TODO: Modify model to return intermediate activations
  activation_data = {
      'logits': np.array(logits),
      'labels': np.array(batch['label']),
      'inputs_rgb_shape': batch['inputs']['rgb'].shape,
      'inputs_spec_shape': batch['inputs']['spectrogram'].shape,
  }
  
  return activation_data


def run_inference_analysis(
    *,
    rng: jnp.ndarray,
    config,
    model_cls,
    dataset,
    workdir: str,
    writer
):
  """Run inference and extract activations.
  
  Args:
    rng: Random number generator
    config: Configuration dict
    model_cls: Model class
    dataset: Dataset object
    workdir: Working directory
    writer: Metrics writer
  """
  logging.info('Starting activation extraction inference...')
  
  # Create output directory
  output_dir = config.get('output_dir', 'analysis_outputs')
  os.makedirs(output_dir, exist_ok=True)
  
  # Build model
  model = model_cls(config, dataset.meta_data)
  
  # Initialize model
  rng, init_rng = jax.random.split(rng)
  input_shapes = dataset.meta_data['input_shape']
  input_dtype = dataset.meta_data.get('input_dtype', jnp.float32)
  
  if isinstance(input_shapes, dict):
    input_spec = {
        modality: (input_shapes[modality], input_dtype)
        for modality in input_shapes
    }
  else:
    input_spec = [(input_shapes, input_dtype)]
  
  # Import initialization function
  from scenic.projects.mbt import train_utils as mbt_train_utils
  
  (params, model_state, num_trainable_params,
   gflops) = mbt_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)
  
  logging.info(f'Model initialized: {num_trainable_params:,} parameters, {gflops} GFLOPs')
  
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
  
  # Load checkpoint if specified
  if config.get('init_from') and config.init_from.get('checkpoint_path'):
    checkpoint_path = config.init_from.checkpoint_path
    if os.path.exists(checkpoint_path):
      logging.info(f'Loading checkpoint from {checkpoint_path}')
      train_state, _ = train_utils.restore_checkpoint(
          checkpoint_path, train_state)
    else:
      logging.warning(f'Checkpoint path {checkpoint_path} not found, using random initialization')
  
  # Don't replicate for single device inference
  # train_state = jax_utils.replicate(train_state)
  
  logging.info('Processing test samples...')
  all_activations = []
  
  # Process each test sample
  for idx, batch in enumerate(dataset.valid_iter):
    if idx >= config.dataset_configs.examples_per_subset['test']:
      break
      
    logging.info(f'Processing sample {idx + 1}/9...')
    
    # Extract activations
    activation_data = extract_activations_from_model(
        model, train_state, batch,
        save_path=os.path.join(output_dir, f'sample_{idx:04d}.pkl')
    )
    
    # Add batch index
    activation_data['batch_idx'] = idx
    all_activations.append(activation_data)
    
    # Save individual sample
    save_path = os.path.join(output_dir, f'sample_{idx:04d}.pkl')
    with open(save_path, 'wb') as f:
      pickle.dump(activation_data, f)
    logging.info(f'Saved activations to {save_path}')
  
  # Save combined activations
  combined_path = os.path.join(output_dir, 'all_activations.pkl')
  with open(combined_path, 'wb') as f:
    pickle.dump(all_activations, f)
  logging.info(f'Saved combined activations to {combined_path}')
  
  logging.info(f'Activation extraction complete! Processed {len(all_activations)} samples.')
  logging.info(f'Outputs saved to: {output_dir}')
  
  return train_state, {}, {}


def main(argv):
  """Main function to run activation analysis."""
  del argv  # Unused
  
  # Use the scenic app framework but with our custom train function
  config = FLAGS.config
  workdir = FLAGS.workdir
  
  # Replace the train function with our inference function
  from scenic.projects.mbt import main as mbt_main
  original_trainer = mbt_main.train
  
  # Monkey patch to use our inference function
  import sys
  sys.modules['scenic.projects.mbt.trainer'].train = run_inference_analysis
  
  # Run the main app
  scenic_app.run(main=mbt_main.main)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
