#!/usr/bin/env python3
"""Debug script to verify model is using loaded checkpoint parameters."""

import sys
sys.path.insert(0, '.')
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import ml_collections
import importlib.util

# Load config
config_path = 'scenic/projects/mbt/configs/audioset/Inference_config.py'
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.get_config()

# Load checkpoint
checkpoint_path = 'CheckPoints/MINI_AV/mbtb32_as-mini_rgb-spec'
print(f"Loading checkpoint: {checkpoint_path}")
ckpt = checkpoints.restore_checkpoint(checkpoint_path, None)
params = ckpt['optimizer']['target']

# Create model
from scenic.projects.mbt import model as mbt_model

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

print("\n" + "="*80)
print("PARAMETER VERIFICATION")
print("="*80)

# Check output projection bias from checkpoint
checkpoint_bias = params['output_projection']['bias']
print(f"\nCheckpoint output_projection bias:")
print(f"  Shape: {checkpoint_bias.shape}")
print(f"  First 5 values: {checkpoint_bias[:5]}")
print(f"  Mean: {np.mean(checkpoint_bias):.6f}")
print(f"  Std: {np.std(checkpoint_bias):.6f}")

# Create dummy inputs
print(f"\nCreating dummy inputs...")
rng = jax.random.PRNGKey(42)
dummy_rgb = jax.random.normal(rng, (1, 1, config.dataset_configs.num_frames, 224, 224, 3))
dummy_spec = jax.random.normal(rng, (1, 1, spec_time_dim, config.dataset_configs.spec_shape[1], 3))
dummy_inputs = {'rgb': dummy_rgb, 'spectrogram': dummy_spec}

# Forward pass with loaded params
print(f"\nRunning forward pass with LOADED parameters...")
variables = {'params': params}
output = model_instance.flax_model.apply(variables, dummy_inputs, train=False, mutable=False)
output_np = np.array(output[0, 0])  # Remove batch and crop dimensions

print(f"  Output shape: {output_np.shape}")
print(f"  Output range: [{output_np.min():.4f}, {output_np.max():.4f}]")
print(f"  Output mean: {output_np.mean():.4f}")
print(f"  First 5 logits: {output_np[:5]}")
print(f"  All negative? {np.all(output_np < 0)}")

# Now try with RANDOM initialization to compare
print(f"\nRunning forward pass with RANDOM parameters (for comparison)...")
# Initialize random params
rng_init = jax.random.PRNGKey(0)
random_params = model_instance.flax_model.init(rng_init, dummy_inputs, train=False)['params']
variables_random = {'params': random_params}
output_random = model_instance.flax_model.apply(variables_random, dummy_inputs, train=False, mutable=False)
output_random_np = np.array(output_random[0, 0])

print(f"  Output range: [{output_random_np.min():.4f}, {output_random_np.max():.4f}]")
print(f"  Output mean: {output_random_np.mean():.4f}")
print(f"  First 5 logits: {output_random_np[:5]}")

# Check if outputs are different (proves we're using loaded params)
diff = np.abs(output_np - output_random_np).mean()
print(f"\nDifference between loaded and random params:")
print(f"  Mean absolute difference: {diff:.6f}")
if diff > 0.01:
    print(f"  ✓ Parameters ARE being used (outputs differ significantly)")
else:
    print(f"  ✗ WARNING: Outputs too similar! Loaded params might NOT be used!")

print("\n" + "="*80)
print("CHECKPOINT VS CONFIG VERIFICATION")
print("="*80)

# Check if patch size matches
checkpoint_patch_h = params['embedding']['kernel'].shape[1]
checkpoint_patch_w = params['embedding']['kernel'].shape[2]
config_patch_h = config.model.patches.size[0]
config_patch_w = config.model.patches.size[1]

print(f"\nPatch size:")
print(f"  Checkpoint: {checkpoint_patch_h}x{checkpoint_patch_w}")
print(f"  Config: {config_patch_h}x{config_patch_w}")
if checkpoint_patch_h == config_patch_h and checkpoint_patch_w == config_patch_w:
    print(f"  ✓ MATCH")
else:
    print(f"  ✗ MISMATCH - This will cause errors!")

# Check classifier type
print(f"\nClassifier type:")
print(f"  Config: {config.model.classifier}")
print(f"  Expected: 'token' (from checkpoint training)")

print("="*80)
