#!/usr/bin/env python3
"""
Quick script to verify that sample files have ground truth labels.
"""

import numpy as np
import sys
import os

def check_sample_files(activation_dir, num_to_check=5):
    """Check if sample files contain ground truth labels."""
    
    print(f"Checking sample files in: {activation_dir}")
    
    # Find sample files
    import glob
    sample_files = sorted(glob.glob(os.path.join(activation_dir, 'sample_*.npz')))
    
    if not sample_files:
        print(f"ERROR: No sample files found in {activation_dir}")
        return False
    
    print(f"Found {len(sample_files)} sample files")
    
    # Check first few samples
    has_labels = True
    num_to_check = min(num_to_check, len(sample_files))
    
    for i, sample_file in enumerate(sample_files[:num_to_check]):
        data = np.load(sample_file)
        keys = list(data.keys())
        
        has_label = 'label' in keys
        has_logits = 'logits' in keys
        
        print(f"\nSample {i} ({os.path.basename(sample_file)}):")
        print(f"  Has 'label': {has_label}")
        print(f"  Has 'logits': {has_logits}")
        
        if has_label:
            label = data['label']
            print(f"  Label shape: {label.shape}")
            # Convert one-hot to index
            if label.ndim > 1:
                label = label.squeeze()
            if label.ndim > 0 and label.shape[0] > 1:
                label_idx = np.argmax(label)
                print(f"  Label index: {label_idx}")
            else:
                print(f"  Label value: {label}")
        else:
            has_labels = False
            print("  WARNING: No ground truth label found!")
        
        if has_logits:
            logits = data['logits']
            print(f"  Logits shape: {logits.shape}")
            pred_idx = np.argmax(logits.squeeze())
            print(f"  Predicted class: {pred_idx}")
    
    print("\n" + "="*80)
    if has_labels:
        print("✓ All checked samples have ground truth labels")
        print("  RDM analysis will use correct class labels")
    else:
        print("✗ Some samples missing ground truth labels")
        print("  Need to re-extract activations with updated script")
    print("="*80)
    
    return has_labels


if __name__ == '__main__':
    activation_dir = sys.argv[1] if len(sys.argv) > 1 else 'audioset_analysis_AV'
    check_sample_files(activation_dir)
