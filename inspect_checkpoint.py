#!/usr/bin/env python3
"""Inspect checkpoint to determine actual patch size."""

import sys
sys.path.insert(0, '.')
from flax.training import checkpoints
import numpy as np

print("="*80)
print("CHECKPOINT INSPECTION")
print("="*80)

checkpoint_path = 'CheckPoints/MINI_AV/mbtb32_as-mini_rgb-spec'
print(f"\nLoading checkpoint: {checkpoint_path}")

ckpt = checkpoints.restore_checkpoint(checkpoint_path, None)
params = ckpt['optimizer']['target']

print("\n" + "="*80)
print("PARAMETER STRUCTURE EXPLORATION")
print("="*80)

def explore_nested_dict(d, prefix='', max_depth=5, current_depth=0):
    """Recursively explore nested dictionary structure."""
    if current_depth >= max_depth:
        return
    
    if isinstance(d, dict):
        for key in sorted(d.keys()):
            value = d[key]
            if isinstance(value, dict):
                print(f"{prefix}{key}/ (dict with {len(value)} keys)")
                explore_nested_dict(value, prefix + "  ", max_depth, current_depth + 1)
            elif hasattr(value, 'shape'):
                print(f"{prefix}{key}: array shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")

print("\nFull parameter structure:")
explore_nested_dict(params, max_depth=4)

print("\n" + "="*80)
print("EMBEDDING LAYER ANALYSIS")
print("="*80)

# Try different possible structures
if 'embedding' in params:
    print("\nFound 'embedding' key. Exploring structure...")
    
    # Check if it's modality-keyed
    if isinstance(params['embedding'], dict):
        emb_keys = list(params['embedding'].keys())
        print(f"Embedding keys: {emb_keys}")
        
        # Look for rgb and spectrogram embeddings
        for modality_name in ['rgb', 'spectrogram', 'RGB', 'Spectrogram']:
            if modality_name in params['embedding']:
                print(f"\n{modality_name.upper()} embedding found:")
                modality_emb = params['embedding'][modality_name]
                if isinstance(modality_emb, dict) and 'kernel' in modality_emb:
                    kernel = modality_emb['kernel']
                    print(f"  Kernel shape: {kernel.shape}")
                    if modality_name.lower() == 'rgb':
                        print(f"  -> RGB patch size: {kernel.shape[0]}x{kernel.shape[1]}, temporal: {kernel.shape[2]}")
                        print(f"  -> This is ViT-B/{kernel.shape[0]}")
else:
    print("No 'embedding' key at top level. Searching deeper...")
    
    # Search for embedding kernels in nested structure
    def find_embedding_kernels(d, path=''):
        """Find all embedding kernel arrays in nested structure."""
        if isinstance(d, dict):
            for key, value in d.items():
                new_path = f"{path}/{key}" if path else key
                if key == 'kernel' and hasattr(value, 'shape') and len(value.shape) >= 4:
                    print(f"\nFound embedding kernel at: {path}")
                    print(f"  Shape: {value.shape}")
                    if 'rgb' in path.lower():
                        print(f"  -> RGB patch size: {value.shape[0]}x{value.shape[1]}")
                elif isinstance(value, dict):
                    find_embedding_kernels(value, new_path)
    
    find_embedding_kernels(params)

print("\n" + "="*80)
print("POSITIONAL EMBEDDING ANALYSIS")
print("="*80)

# Check for positional embeddings to infer spatial resolution
for modality in ['rgb', 'spectrogram']:
    key_variants = [
        f'Transformer_{modality}/posembed_input/pos_embedding',
        f'Transformer/posembed_input_{modality}/pos_embedding',
        f'pos_embedding_{modality}',
    ]
    
    found = False
    for key in key_variants:
        # Navigate nested dict structure
        parts = key.split('/')
        current = params
        try:
            for part in parts:
                if part in current:
                    current = current[part]
                else:
                    break
            else:
                # Successfully navigated
                if hasattr(current, 'shape'):
                    print(f"\n{modality.upper()} positional embedding:")
                    print(f"  Shape: {current.shape}")
                    # Shape is typically [1, num_patches + 1, hidden_dim]
                    # For 224x224 image with 16x16 patches: (224/16)^2 = 196 patches
                    # For 224x224 image with 32x32 patches: (224/32)^2 = 49 patches
                    if len(current.shape) >= 2:
                        num_positions = current.shape[1] - 1  # Subtract CLS token
                        print(f"  Number of spatial positions: {num_positions}")
                        if modality == 'rgb':
                            # Calculate expected patch size
                            import math
                            side_length = int(math.sqrt(num_positions))
                            if side_length * side_length == num_positions:
                                inferred_patch_size = 224 // side_length
                                print(f"  Inferred patch size (224/{side_length}): {inferred_patch_size}x{inferred_patch_size}")
                    found = True
                    break
        except (KeyError, TypeError):
            continue
    
    if not found:
        print(f"\n{modality.upper()}: Could not find positional embedding")

print("\n" + "="*80)
print("MODEL_STATE INSPECTION (CRITICAL FOR BATCH NORM)")
print("="*80)

if 'model_state' in ckpt:
    print("\n✓ Found model_state in checkpoint!")
    model_state = ckpt['model_state']
    
    if isinstance(model_state, dict):
        print(f"  model_state keys: {list(model_state.keys())}")
        
        # Check for batch_stats (batch normalization statistics)
        if 'batch_stats' in model_state:
            print(f"\n  ✓ Found batch_stats (batch normalization statistics)")
            batch_stats = model_state['batch_stats']
            print(f"  batch_stats type: {type(batch_stats)}")
            if isinstance(batch_stats, dict):
                print(f"  batch_stats keys: {list(batch_stats.keys())[:10]}")
        else:
            print(f"\n  ✗ No batch_stats found in model_state")
            print(f"  Available keys: {list(model_state.keys())}")
    else:
        print(f"  model_state type: {type(model_state)}")
else:
    print("\n✗ WARNING: No model_state in checkpoint!")
    print("  This could cause incorrect predictions if model uses batch normalization!")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\nBased on the embedding kernel shape, this checkpoint was trained with:")
if 'embedding' in params:
    if 'kernel' in params['embedding']:
        rgb_kernel = params['embedding']['kernel']
        patch_h, patch_w = rgb_kernel.shape[0], rgb_kernel.shape[1]
        print(f"  Patch size: {patch_h}x{patch_w}")
        print(f"  Architecture: ViT-B/{patch_h}")
        print(f"\nYour Inference_config.py should have:")
        print(f"  config.model.patches.size = [{patch_h}, {patch_w}, 2]")

print("\n⚠️  CRITICAL: Check if model_state/batch_stats is being loaded!")
print("="*80)
