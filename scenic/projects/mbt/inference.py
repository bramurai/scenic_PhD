"""Inference script for MBT model with activation extraction."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, List
import numpy as np
from absl import logging

from scenic.projects.mbt import model as mbt_model
from scenic.train_lib import train_utils
import ml_collections


def prepare_inference_config(base_config_path: str) -> ml_collections.ConfigDict:
    """Prepare configuration for inference with activation extraction.
    
    Args:
        base_config_path: Path to base config file
        
    Returns:
        Modified config optimized for inference
    """
    # Import the config module
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", base_config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.get_config()
    
    # Optimize for inference
    config.batch_size = 1  # Process one sample at a time for detailed analysis
    config.model.dropout_rate = 0.0  # No dropout during inference
    config.model.attention_dropout_rate = 0.0
    config.model.stochastic_droplayer_rate = 0.0
    
    # Enable returning intermediate representations
    config.model.return_preclassifier = False  # Set to True to get all token embeddings
    config.model.return_prelogits = False  # Set to True to get pre-classification features
    
    # Disable training-specific features
    config.dataset_configs.spec_augment = False
    config.dataset_configs.augmentation_params.do_color_augment = False
    config.dataset_configs.augmentation_params.prob_scale_jitter = 0.0
    
    # Use float16 for memory efficiency (optional)
    # config.model_dtype_str = 'float16'
    
    logging.info("Inference config prepared:")
    logging.info(f"  Batch size: {config.batch_size}")
    logging.info(f"  Dropout disabled: {config.model.dropout_rate == 0.0}")
    logging.info(f"  Model dtype: {config.model_dtype_str}")
    
    return config


def build_dataset_metadata(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Build dataset metadata required for model initialization.
    
    Args:
        config: Configuration dict
        
    Returns:
        Dataset metadata dictionary
    """
    return {
        'num_classes': config.dataset_configs.num_classes,
        'num_train_examples': config.dataset_configs.examples_per_subset['train'],
        'num_eval_examples': config.dataset_configs.examples_per_subset['validation'],
        'input_dtype': getattr(jnp, config.data_dtype_str),
        'target_is_onehot': config.dataset_configs.one_hot_labels,
        'input_shape': {
            'rgb': (-1, config.dataset_configs.num_frames, 
                   config.dataset_configs.crop_size, 
                   config.dataset_configs.crop_size, 3),
            'spectrogram': (-1, 
                          config.dataset_configs.spec_shape[0] * config.dataset_configs.num_spec_frames,
                          config.dataset_configs.spec_shape[1], 3)
        }
    }


def extract_activations_with_intermediates(
    model,
    variables: Dict,
    input_data: Dict[str, jnp.ndarray],
    layers_to_extract: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract activations from specific layers during forward pass.
    
    This uses Flax's mutable='intermediates' to capture layer outputs.
    
    Args:
        model: Flax model
        variables: Model parameters
        input_data: Input batch {'rgb': ..., 'spectrogram': ...}
        layers_to_extract: List of layer names to extract (None = all)
        
    Returns:
        Dictionary containing:
            - 'outputs': Final model predictions
            - 'intermediates': Dict of intermediate activations
    """
    # Run model with intermediate capture
    outputs, state = model.apply(
        variables,
        input_data,
        train=False,
        debug=False,
        mutable=['intermediates'],  # Capture intermediate values
        capture_intermediates=True   # Flax feature to save layer outputs
    )
    
    intermediates = state.get('intermediates', {})
    
    result = {
        'outputs': outputs,
        'intermediates': intermediates,
    }
    
    return result


def extract_encoder_layer_activations(
    model,
    variables: Dict,
    input_data: Dict[str, jnp.ndarray],
    target_layers: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Extract activations from specific transformer encoder layers.
    
    Args:
        model: Flax model
        variables: Model parameters
        input_data: Input batch
        target_layers: List of layer indices to extract (e.g., [0, 5, 11])
                      None = extract all layers
        
    Returns:
        Dictionary with layer-wise activations
    """
    # For MBT, we need to modify the model to return intermediate outputs
    # This requires setting return_preclassifier=True
    
    # Get all token embeddings before classification
    outputs = model.apply(
        variables,
        input_data,
        train=False,
        debug=False,
        mutable=False
    )
    
    return {
        'final_output': outputs,
        'note': 'For detailed layer-wise extraction, model needs modification'
    }


def extract_attention_patterns(
    model,
    variables: Dict,
    input_data: Dict[str, jnp.ndarray]
) -> Dict[str, Any]:
    """Extract attention patterns from the model.
    
    Note: This requires modifying the model to return attention weights.
    """
    # This would require model modifications to return attention weights
    # from MultiHeadDotProductAttention layers
    raise NotImplementedError(
        "Attention extraction requires modifying the model's "
        "MultiHeadDotProductAttention to return attention weights. "
        "See Flax documentation on 'decode' mode for self-attention."
    )


def run_inference(
    config: ml_collections.ConfigDict,
    checkpoint_path: str,
    input_data: Dict[str, jnp.ndarray],
    extract_intermediates: bool = False
) -> Dict[str, Any]:
    """Run inference on pretrained MBT model with optional activation extraction.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to checkpoint file (can be file or directory)
        input_data: Input batch with keys 'rgb' and/or 'spectrogram'
        extract_intermediates: Whether to extract intermediate activations
        
    Returns:
        Dictionary containing predictions and optionally intermediate activations
    """
    import os
    from flax import serialization
    
    # Build dataset metadata
    dataset_meta_data = build_dataset_metadata(config)
    
    # Build model
    model_cls = mbt_model.MBTMultilabelClassificationModel
    model_instance = model_cls(config, dataset_meta_data)
    flax_model = model_instance.flax_model
    
    # Load checkpoint - handle both directory and file paths
    try:
        # Try standard Scenic checkpoint loading first
        train_state = train_utils.restore_checkpoint(
            checkpoint_path,
            assert_exist=True
        )
    except (ValueError, FileNotFoundError) as e:
        # If that fails, try loading directly as a MessagePack file
        logging.info(f"Standard checkpoint loading failed: {e}")
        logging.info(f"Attempting to load checkpoint as MessagePack file directly...")
        
        # Check if path is a file
        if os.path.isfile(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                train_state = serialization.msgpack_restore(f.read())
            logging.info(f"✓ Successfully loaded checkpoint from {checkpoint_path}")
            logging.info(f"Checkpoint keys: {list(train_state.keys())}")
        else:
            raise ValueError(
                f"Could not load checkpoint from {checkpoint_path}. "
                f"Tried both directory format and direct file loading."
            )
    
    # Handle different checkpoint formats
    if 'params' in train_state:
        params = train_state['params']
        logging.info("Using params from train_state['params']")
    elif 'optimizer' in train_state and 'target' in train_state['optimizer']:
        # Older Flax format with optimizer state
        params = train_state['optimizer']['target']
        logging.info("Using params from optimizer.target")
    else:
        # Assume the entire state is the params
        params = train_state
        logging.info("Using entire checkpoint as params")
    
    variables = {'params': params}
    
    # Run inference
    if extract_intermediates:
        @jax.jit
        def predict_with_intermediates(batch):
            return extract_activations_with_intermediates(
                flax_model, variables, batch
            )
        
        results = predict_with_intermediates(input_data)
    else:
        @jax.jit
        def predict(batch):
            return flax_model.apply(
                variables,
                batch,
                train=False,
                mutable=False
            )
        
        predictions = predict(input_data)
        results = {'outputs': predictions}
    
    return results


def analyze_activations(
    activations: Dict[str, Any],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze extracted activations and compute statistics.
    
    Args:
        activations: Dictionary of activations from different layers
        save_path: Optional path to save analysis results
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    for layer_name, activation in activations.items():
        if isinstance(activation, jnp.ndarray):
            analysis[layer_name] = {
                'shape': activation.shape,
                'mean': float(jnp.mean(activation)),
                'std': float(jnp.std(activation)),
                'min': float(jnp.min(activation)),
                'max': float(jnp.max(activation)),
                'sparsity': float(jnp.mean(activation == 0)),
            }
            
            logging.info(f"\n{layer_name}:")
            logging.info(f"  Shape: {activation.shape}")
            logging.info(f"  Mean: {analysis[layer_name]['mean']:.4f}")
            logging.info(f"  Std: {analysis[layer_name]['std']:.4f}")
            logging.info(f"  Sparsity: {analysis[layer_name]['sparsity']:.2%}")
    
    if save_path:
        np.save(save_path, analysis)
        logging.info(f"Analysis saved to {save_path}")
    
    return analysis


# Example usage
def main():
    """Example of how to use the inference script."""
    
    # 1. Prepare config
    config = prepare_inference_config(
        'scenic/projects/mbt/configs/audioset/balanced_audioset_base.py'
    )
    
    # 2. Create dummy input for testing
    dummy_input = {
        'rgb': jnp.zeros((1, 32, 224, 224, 3)),
        'spectrogram': jnp.zeros((1, 800, 128, 3))
    }
    
    # 3. Run inference
    checkpoint_path = 'path/to/checkpoint'
    results = run_inference(
        config,
        checkpoint_path,
        dummy_input,
        extract_intermediates=True
    )
    
    # 4. Analyze activations
    if 'intermediates' in results:
        analysis = analyze_activations(results['intermediates'])
    
    return results


if __name__ == '__main__':
    main()