#!/usr/bin/env python3
"""Recompute mAP from saved logits."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Load saved logits and labels
print("Loading logits and labels...")
data = np.load('audioset_analysis_All_date/all_logits.npz')
logits = data['logits']
labels = data['labels']

print(f"Logits shape: {logits.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Number of samples: {logits.shape[0]}")
print(f"Number of classes: {logits.shape[1]}")

# Load class names
labels_df = pd.read_csv('Video_csvs/audioset_labels.csv')
index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
num_classes = len(labels_df)

# Apply sigmoid to get probabilities
print("\nApplying sigmoid to logits...")
probs = 1.0 / (1.0 + np.exp(-logits))

# Compute per-class AP
print("Computing per-class Average Precision...")
aps_per_class = {}
for class_idx in range(num_classes):
    y_true = labels[:, class_idx]
    y_score = probs[:, class_idx]
    
    # Only compute AP if there are positive samples
    if y_true.sum() > 0:
        ap = average_precision_score(y_true, y_score)
        aps_per_class[class_idx] = ap

# Compute mAP
map_score = np.mean(list(aps_per_class.values())) if aps_per_class else 0.0

print("\n" + "="*80)
print("RESULTS:")
print("="*80)
print(f"mAP: {map_score:.4f} ({map_score*100:.2f}%)")
print(f"Number of classes with samples: {len(aps_per_class)}/{num_classes}")

# Show top and bottom classes by AP
ap_with_names = [(index_to_name[class_idx], ap, class_idx) 
                 for class_idx, ap in aps_per_class.items()]
ap_with_names.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 classes by AP:")
for name, ap, idx in ap_with_names[:10]:
    num_positive = int(labels[:, idx].sum())
    print(f"  {name:40s}: {ap:.4f} ({num_positive} samples)")

print("\nBottom 10 classes by AP:")
for name, ap, idx in ap_with_names[-10:]:
    num_positive = int(labels[:, idx].sum())
    print(f"  {name:40s}: {ap:.4f} ({num_positive} samples)")

# Save detailed results
print("\nSaving detailed results...")
results = []
for class_idx, ap in aps_per_class.items():
    num_positive = int(labels[:, class_idx].sum())
    results.append({
        'class_idx': class_idx,
        'class_name': index_to_name[class_idx],
        'ap': ap,
        'num_samples': num_positive
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ap', ascending=False)
results_df.to_csv('audioset_analysis_All_date/per_class_ap.csv', index=False)
print(f"Saved per-class AP to audioset_analysis_All_date/per_class_ap.csv")

print("\n" + "="*80)
