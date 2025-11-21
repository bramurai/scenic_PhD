#!/usr/bin/env python3
"""Display mAP metrics from summary.npz file."""

import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python show_map.py <path_to_summary.npz>")
    print("Example: python show_map.py audioset_analysis_AV/summary.npz")
    sys.exit(1)

summary_path = sys.argv[1]

print(f"\nLoading summary from: {summary_path}")
summary = np.load(summary_path)

print("\nAvailable keys:", list(summary.keys()))

if 'micro_map' in summary and 'macro_map' in summary:
    print("\n" + "="*80)
    print("AudioSet Evaluation Metrics")
    print("="*80)
    print(f"Micro mAP (across all samples): {summary['micro_map']:.4f}")
    print(f"Macro mAP (average per class):  {summary['macro_map']:.4f}")
    print(f"Number of samples:              {summary['num_samples']}")
    
    if 'per_class_ap' in summary:
        per_class_ap = summary['per_class_ap']
        valid_aps = per_class_ap[~np.isnan(per_class_ap)]
        print(f"Number of classes evaluated:    {len(valid_aps)}")
        print(f"\nPer-class AP statistics:")
        print(f"  Min:    {valid_aps.min():.4f}")
        print(f"  Max:    {valid_aps.max():.4f}")
        print(f"  Mean:   {valid_aps.mean():.4f}")
        print(f"  Median: {np.median(valid_aps):.4f}")
        print(f"  Std:    {valid_aps.std():.4f}")
        
        # Show top 10 classes
        print(f"\nTop 10 classes by AP:")
        top_indices = np.argsort(per_class_ap)[::-1][:10]
        for i, idx in enumerate(top_indices, 1):
            if not np.isnan(per_class_ap[idx]):
                print(f"  {i}. Class {idx}: {per_class_ap[idx]:.4f}")
        
        # Show bottom 10 classes
        print(f"\nBottom 10 classes by AP:")
        bottom_indices = np.argsort(per_class_ap)[:10]
        for i, idx in enumerate(bottom_indices, 1):
            if not np.isnan(per_class_ap[idx]):
                print(f"  {i}. Class {idx}: {per_class_ap[idx]:.4f}")
    
    print("="*80)
    print("\nNOTE: If TFRecords have incorrect label mapping (custom 0-76 instead of")
    print("AudioSet 0-526), the mAP will be INCORRECT. Expected mAP for AudioSet: ~50%")
    print("="*80)
else:
    print("\nNo mAP metrics found in summary. Available keys:", list(summary.keys()))
    if 'logits' in summary and 'labels' in summary:
        print("\nSummary contains logits and labels but no computed metrics.")
        print("Run extract_mbt_activations.py again to compute mAP.")
