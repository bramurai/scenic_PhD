#!/usr/bin/env python3
"""Quick script to show what classes are in your RDM analysis."""

import pickle
import sys

summary_path = 'audioset_rdm_analysis_with_labels/summary.pkl'

try:
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
    
    print(f"Total samples: {summary['num_samples']}")
    print(f"Total unique classes: {summary['num_classes']}")
    print(f"Distance metric: {summary['distance_metric']}")
    print(f"Standardized: {summary['standardized']}")
    print(f"\nClasses in your dataset:")
    print("=" * 60)
    
    label_to_name = summary['label_to_name']
    unique_labels = summary['unique_labels']
    
    for label in sorted(unique_labels):
        name = label_to_name.get(label, f'Unknown class {label}')
        print(f"  {label:3d}: {name}")
    
except FileNotFoundError:
    print(f"Error: {summary_path} not found.")
    print("Run compute_rdm.py first to generate the summary.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading summary: {e}")
    sys.exit(1)
