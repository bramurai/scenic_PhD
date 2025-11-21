#!/usr/bin/env python3
"""Verify that predicted labels match expected AudioSet structure."""

import numpy as np
import glob
import os

# Load a few samples and check their predictions
sample_files = sorted(glob.glob('audioset_analysis_AV/sample_*.npz'))[:10]

print("Checking first 10 samples:")
print("=" * 80)

for sample_file in sample_files:
    data = np.load(sample_file)
    logits = data['logits']
    
    # Remove batch dimension if present
    if logits.ndim > 1 and logits.shape[0] == 1:
        logits = logits[0]
    
    # Get top 5 predictions
    top5_indices = np.argsort(logits)[-5:][::-1]
    top5_probs = logits[top5_indices]
    
    print(f"\n{os.path.basename(sample_file)}:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Max logit value: {np.max(logits):.3f}")
    print(f"  Top 5 predictions:")
    for idx, logit in zip(top5_indices, top5_probs):
        print(f"    Class {idx:3d}: logit = {logit:.3f}")
    
    # Check if logits are probabilities (sum to 1) or raw scores
    if np.max(logits) <= 1.0 and np.min(logits) >= 0.0:
        total = np.sum(logits)
        print(f"  Sum of all values: {total:.3f} (appears to be probabilities)")
    else:
        print(f"  Min/Max: {np.min(logits):.3f} / {np.max(logits):.3f} (raw logits)")

print("\n" + "=" * 80)
print("IMPORTANT: AudioSet model should output 527 classes")
print(f"Your model outputs: {logits.shape[0]} classes")

if logits.shape[0] != 527:
    print("\n⚠️  WARNING: Expected 527 classes for AudioSet!")
    print("   This might be a VGGSound model (309 classes) or other dataset.")
