#!/usr/bin/env python3
"""Diagnose what's wrong with the logits."""

import numpy as np

# Load logits and labels
print("Loading data...")
data = np.load('audioset_analysis_All_date/all_logits.npz')
logits = data['logits']
labels = data['labels']

print(f"Logits shape: {logits.shape}")
print(f"Labels shape: {labels.shape}")

# Check logits statistics
print("\n" + "="*80)
print("LOGITS STATISTICS:")
print("="*80)
print(f"Min: {logits.min():.4f}")
print(f"Max: {logits.max():.4f}")
print(f"Mean: {logits.mean():.4f}")
print(f"Std: {logits.std():.4f}")

# Check if logits are all zeros or constant
print(f"\nAre all logits zero? {np.all(logits == 0)}")
print(f"Are all logits the same? {np.all(logits == logits[0, 0])}")

# Check a few samples
print("\nFirst sample logits (first 10 values):")
print(logits[0, :10])

print("\nFirst sample logits statistics:")
print(f"  Min: {logits[0].min():.4f}")
print(f"  Max: {logits[0].max():.4f}")
print(f"  Mean: {logits[0].mean():.4f}")
print(f"  Argmax: {logits[0].argmax()}")

# Check labels statistics
print("\n" + "="*80)
print("LABELS STATISTICS:")
print("="*80)
print(f"Min: {labels.min():.4f}")
print(f"Max: {labels.max():.4f}")
print(f"Total positive labels: {labels.sum():.0f}")
print(f"Avg labels per sample: {labels.sum(axis=1).mean():.2f}")

print("\nFirst sample labels (non-zero indices):")
positive_indices = np.where(labels[0] > 0)[0]
print(f"  Positive classes: {positive_indices}")
print(f"  Number of positive classes: {len(positive_indices)}")

# Check if predictions align with labels at all
print("\n" + "="*80)
print("PREDICTION VS LABEL ALIGNMENT:")
print("="*80)

# Apply sigmoid
probs = 1.0 / (1.0 + np.exp(-logits))
print(f"Probabilities - Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")

# For first sample, check if top predictions match labels
sample_idx = 0
sample_probs = probs[sample_idx]
sample_labels = labels[sample_idx]

top_k = 10
top_indices = np.argsort(sample_probs)[-top_k:][::-1]
print(f"\nFirst sample - Top {top_k} predicted classes (by probability):")
for rank, idx in enumerate(top_indices, 1):
    is_correct = "✓" if sample_labels[idx] > 0 else "✗"
    print(f"  {rank}. Class {idx}: prob={sample_probs[idx]:.4f} {is_correct}")

print(f"\nFirst sample - Actual positive classes:")
for idx in positive_indices[:10]:
    print(f"  Class {idx}: prob={sample_probs[idx]:.4f}")

# Check overall prediction quality
print("\n" + "="*80)
print("OVERALL PREDICTION QUALITY:")
print("="*80)

# For each sample, check if any of top-5 predictions are correct
top5_correct = 0
for i in range(len(logits)):
    top5_idx = np.argsort(probs[i])[-5:]
    if np.any(labels[i, top5_idx] > 0):
        top5_correct += 1

print(f"Samples where at least one top-5 prediction is correct: {top5_correct}/{len(logits)} ({100*top5_correct/len(logits):.1f}%)")

# Check if logits are in reasonable range
print("\n" + "="*80)
print("LOGITS RANGE CHECK:")
print("="*80)
print("Typical trained model logits should be in range [-10, 10]")
print(f"Your logits range: [{logits.min():.2f}, {logits.max():.2f}]")

if logits.max() < 1.0 and logits.min() > -1.0:
    print("\n⚠️  WARNING: Logits are in a very small range!")
    print("   This suggests the model might not be properly trained or loaded.")
    print("   Expected range for trained model: roughly [-10, 10]")
    print("   Your range: roughly [{:.2f}, {:.2f}]".format(logits.min(), logits.max()))
    print("\n   Possible issues:")
    print("   1. Wrong checkpoint loaded")
    print("   2. Model architecture mismatch")
    print("   3. Checkpoint is from early in training (not converged)")
    print("   4. Wrong output layer being extracted")
