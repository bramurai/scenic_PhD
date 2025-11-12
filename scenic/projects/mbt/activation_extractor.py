"""Model wrapper for extracting layer-wise activations and attention weights.

This module provides functionality to:
1. Capture intermediate layer outputs during forward pass
2. Extract attention weights from all attention layers
3. Save activations for downstream analysis
"""

import functools
from typing import Any, Dict, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils
import pickle
import os


class ActivationExtractor:
  """Wrapper to extract activations and attention weights from MBT model."""
  
  def __init__(self, model, config):
    """Initialize the activation extractor.
    
    Args:
      model: The MBT model instance
      config: Model configuration
    """
    self.model = model
    self.config = config
    self.activations = {}
    self.attention_weights = {}
    
  def extract_from_batch(
      self,
      train_state,
      batch: Dict[str, jnp.ndarray],
      save_dir: str = 'analysis_outputs'
  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract activations and attention weights for a single batch.
    
    Args:
      train_state: Current training state with model parameters
      batch: Input batch with 'inputs' and 'label' keys
      save_dir: Directory to save extracted features
      
    Returns:
      Tuple of (activations_dict, attention_weights_dict)
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run forward pass with intermediate outputs
    variables = {
        'params': train_state.params,
        **train_state.model_state
    }
    
    # Forward pass in eval mode (no dropout, etc.)
    output, intermediates = self.model.flax_model.apply(
        variables,
        batch['inputs'],
        train=False,
        mutable=False,
        capture_intermediates=True  # This captures all intermediate values
    )
    
    # Extract activations from intermediates
    activations = self._parse_intermediates(intermediates)
    
    # Save to disk
    batch_idx = batch.get('batch_idx', 0)
    self._save_activations(activations, batch_idx, save_dir)
    
    return activations, output
  
  def _parse_intermediates(self, intermediates: Dict) -> Dict[str, Any]:
    """Parse intermediate activations from model.
    
    Args:
      intermediates: Dictionary of intermediate values from Flax
      
    Returns:
      Organized dictionary of layer activations
    """
    activations = {}
    
    # The intermediates will contain outputs from each layer
    # Format depends on how Flax captured them
    if 'intermediates' in intermediates:
      for key, value in intermediates['intermediates'].items():
        # Convert to numpy for easier handling
        if hasattr(value, 'shape'):
          activations[key] = np.array(value)
    
    return activations
  
  def _save_activations(
      self,
      activations: Dict[str, Any],
      batch_idx: int,
      save_dir: str
  ):
    """Save activations to disk.
    
    Args:
      activations: Dictionary of layer activations
      batch_idx: Index of current batch
      save_dir: Directory to save to
    """
    filename = os.path.join(save_dir, f'activations_batch_{batch_idx:04d}.pkl')
    with open(filename, 'wb') as f:
      pickle.dump(activations, f)
    print(f'Saved activations to {filename}')


def extract_attention_weights_from_encoder(
    encoder_module,
    x: jnp.ndarray,
    params: Dict,
    deterministic: bool = True
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
  """Extract attention weights from encoder layers.
  
  This is a custom function to manually extract attention weights
  by calling the encoder with modified attention layers.
  
  Args:
    encoder_module: The encoder module
    x: Input tensor
    params: Model parameters
    deterministic: Whether to use deterministic mode
    
  Returns:
    Tuple of (output, list of attention weight matrices)
  """
  attention_weights = []
  
  # We need to modify this based on the actual MBT encoder structure
  # For now, return placeholder
  # TODO: Implement actual attention weight extraction
  
  return x, attention_weights


def create_activation_extraction_model(model_cls, config):
  """Create a model wrapper for activation extraction.
  
  Args:
    model_cls: Model class to wrap
    config: Model configuration
    
  Returns:
    Model instance with activation extraction capabilities
  """
  # Create base model
  model = model_cls(config, {
      'num_classes': config.dataset_configs.num_classes,
      'input_shape': {
          'rgb': (-1, config.dataset_configs.num_frames, 224, 224, 3),
          'spectrogram': (-1, config.dataset_configs.num_spec_frames, 128, 3)
      },
      'input_dtype': jnp.float32,
      'target_is_onehot': True
  })
  
  # Wrap with activation extractor
  extractor = ActivationExtractor(model, config)
  
  return model, extractor
