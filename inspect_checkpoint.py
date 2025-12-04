#!/usr/bin/env python3
"""Inspect MBT checkpoint to verify it's loaded correctly."""

import pickle
import numpy as np
from flax.training import checkpoints

checkpoint_path = 'CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec'

print("Loading checkpoint...")
checkpoint = checkpoints.restore_checkpoint(checkpoint_path, None)

print("\n" + "="*80)
print("CHECKPOINT STRUCTURE:")
print("="*80)
print(f"Top-level keys: {list(checkpoint.keys())}")

# Check if it has params
if 'params' in checkpoint:
    params = checkpoint['params']
    print("\nFound 'params' key")
elif 'optimizer' in checkpoint and 'target' in checkpoint['optimizer']:
    params = checkpoint['optimizer']['target']
    print("\nFound params in 'optimizer.target'")
else:
    params = checkpoint
    print("\nUsing entire checkpoint as params")

print(f"\nParams type: {type(params)}")
print(f"Params keys: {list(params.keys())[:10]}...")

# Check for classifier head (final layer)
def find_classifier(d, path=''):
    """Recursively find classifier weights."""
    if hasattr(d, 'items'):
        for k, v in d.items():
            new_path = f"{path}/{k}" if path else k
            if 'head' in k.lower() or 'classifier' in k.lower() or 'output_projection' in k.lower():
                if hasattr(v, 'shape'):
                    print(f"\nFound classifier at: {new_path}")
                    print(f"  Shape: {v.shape}")
                    print(f"  Mean: {np.mean(v):.4f}, Std: {np.std(v):.4f}")
                else:
                    print(f"\nFound classifier dict at: {new_path}")
                    find_classifier(v, new_path)
            elif hasattr(v, 'items'):
                find_classifier(v, new_path)

print("\n" + "="*80)
print("SEARCHING FOR CLASSIFIER WEIGHTS:")
print("="*80)
find_classifier(params)

# Check for encoder blocks
def count_encoder_blocks(d, path=''):
    """Count encoder blocks."""
    count = 0
    if hasattr(d, 'items'):
        for k, v in d.items():
            if 'encoderblock_' in k:
                count += 1
            elif hasattr(v, 'items'):
                count += count_encoder_blocks(v, path)
    return count

encoder_count = count_encoder_blocks(params)
print(f"\n" + "="*80)
print(f"MODEL ARCHITECTURE:")
print("="*80)
print(f"Number of encoder blocks found: {encoder_count}")

# Check if there are any trained weights (should not be all zeros)
def check_if_trained(d, max_checks=5):
    """Check a few weight matrices to see if they're trained."""
    checks = []
    count = 0
    
    def recurse(d, path=''):
        nonlocal count
        if count >= max_checks:
            return
        
        if hasattr(d, 'items'):
            for k, v in d.items():
                if count >= max_checks:
                    return
                
                new_path = f"{path}/{k}" if path else k
                
                if hasattr(v, 'shape') and len(v.shape) >= 2 and 'kernel' in k.lower():
                    # Found a weight matrix
                    v_array = np.array(v)
                    checks.append({
                        'path': new_path,
                        'shape': v.shape,
                        'mean': float(np.mean(np.abs(v_array))),
                        'std': float(np.std(v_array)),
                        'max': float(np.max(np.abs(v_array)))
                    })
                    count += 1
                elif hasattr(v, 'items'):
                    recurse(v, new_path)
    
    recurse(d)
    return checks

print("\n" + "="*80)
print("SAMPLE WEIGHT MATRICES:")
print("="*80)
weight_checks = check_if_trained(params, max_checks=5)
for check in weight_checks:
    print(f"\n{check['path']}")
    print(f"  Shape: {check['shape']}")
    print(f"  Mean (abs): {check['mean']:.6f}")
    print(f"  Std: {check['std']:.6f}")
    print(f"  Max (abs): {check['max']:.6f}")

# Check if weights are mostly zeros (untrained)
if weight_checks:
    avg_mean = np.mean([c['mean'] for c in weight_checks])
    if avg_mean < 0.001:
        print("\n⚠️  WARNING: Weights appear to be mostly zeros - model may be untrained!")
    else:
        print(f"\n✓ Weights appear trained (avg mean abs value: {avg_mean:.6f})")

# Check for global_step or other training metadata
print("\n" + "="*80)
print("TRAINING METADATA:")
print("="*80)
if 'global_step' in checkpoint:
    print(f"Global step: {checkpoint['global_step']}")
if 'model_state' in checkpoint:
    print(f"Model state keys: {list(checkpoint['model_state'].keys())}")

print("\n" + "="*80)
print("CHECKPOINT INSPECTION COMPLETE")
print("="*80)
