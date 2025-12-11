#!/usr/bin/env python3
"""Check the output_projection (classifier) layer specifically."""

import numpy as np
from flax.training import checkpoints

checkpoint_path = 'CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec'

print("Loading checkpoint...")
checkpoint = checkpoints.restore_checkpoint(checkpoint_path, None)

params = checkpoint['optimizer']['target']

print("\n" + "="*80)
print("OUTPUT PROJECTION (CLASSIFIER) LAYER:")
print("="*80)

if 'output_projection' in params:
    op = params['output_projection']
    print(f"output_projection keys: {list(op.keys())}")
    
    if 'kernel' in op:
        kernel = np.array(op['kernel'])
        print(f"\nClassifier kernel shape: {kernel.shape}")
        print(f"Expected shape: (hidden_dim, num_classes) = (768, 527)")
        print(f"Actual shape: {kernel.shape}")
        
        if kernel.shape == (768, 527):
            print("✓ Shape matches expected AudioSet classifier!")
        else:
            print("✗ WARNING: Shape doesn't match AudioSet (527 classes)")
        
        print(f"\nClassifier statistics:")
        print(f"  Mean: {np.mean(kernel):.6f}")
        print(f"  Std: {np.std(kernel):.6f}")
        print(f"  Min: {np.min(kernel):.6f}")
        print(f"  Max: {np.max(kernel):.6f}")
        print(f"  Mean (abs): {np.mean(np.abs(kernel)):.6f}")
        
        # Check if weights are reasonable
        if np.mean(np.abs(kernel)) < 0.001:
            print("\n✗ WARNING: Classifier weights are very small - may be untrained!")
        elif np.mean(np.abs(kernel)) > 10:
            print("\n✗ WARNING: Classifier weights are very large - may be corrupted!")
        else:
            print(f"\n✓ Classifier weights appear reasonable")
        
        # Show first few class weight norms
        print("\nFirst 10 class weight norms:")
        for i in range(min(10, kernel.shape[1])):
            class_weights = kernel[:, i]
            norm = np.linalg.norm(class_weights)
            print(f"  Class {i}: L2 norm = {norm:.4f}")
    
    if 'bias' in op:
        bias = np.array(op['bias'])
        print(f"\nClassifier bias shape: {bias.shape}")
        print(f"Bias statistics:")
        print(f"  Mean: {np.mean(bias):.6f}")
        print(f"  Std: {np.std(bias):.6f}")
        print(f"  Min: {np.min(bias):.6f}")
        print(f"  Max: {np.max(bias):.6f}")
        
        # Show first few bias values
        print("\nFirst 10 bias values:")
        for i in range(min(10, len(bias))):
            print(f"  Class {i}: {bias[i]:.4f}")
else:
    print("✗ No 'output_projection' found in params!")
    print(f"Available keys: {list(params.keys())}")

print("\n" + "="*80)
print("Checking global step...")
print("="*80)
if 'global_step' in checkpoint:
    print(f"Training step: {checkpoint['global_step']}")
    print("Note: mbtb32_as-500k checkpoint is from step 500,000")
    print("      Check if this is the final/best checkpoint or an intermediate one")
