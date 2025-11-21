#!/usr/bin/env python3
"""Check prediction confidence and within-class consistency."""

import numpy as np
import pickle

# Load summary
with open('audioset_rdm_analysis/summary.pkl', 'rb') as f:
    summary = pickle.load(f)

# Load one RDM (e.g., final layer)
rdm_file = 'audioset_rdm_analysis/rdm_encoder_block_L11_rgb_output.npz'
data = np.load(rdm_file)
rdm = data['rdm']
labels = data['labels']

label_to_name = summary['label_to_name']
unique_labels = summary['unique_labels']

print("=" * 80)
print("WITHIN-CLASS DISSIMILARITY ANALYSIS")
print("=" * 80)
print("\nChecking how similar samples are WITHIN each class...")
print("(Lower = more consistent/confident, Higher = confused/diverse)\n")

for label in sorted(unique_labels):
    # Get indices of samples belonging to this class
    indices = np.where(labels == label)[0]
    n_samples = len(indices)
    
    if n_samples < 2:
        continue  # Skip classes with only 1 sample
    
    # Extract sub-RDM for this class (within-class dissimilarities)
    sub_rdm = rdm[np.ix_(indices, indices)]
    
    # Get upper triangle (don't count diagonal which is always 0)
    upper_tri_indices = np.triu_indices_from(sub_rdm, k=1)
    within_class_dists = sub_rdm[upper_tri_indices]
    
    # Statistics
    mean_dist = np.mean(within_class_dists)
    std_dist = np.std(within_class_dists)
    min_dist = np.min(within_class_dists)
    max_dist = np.max(within_class_dists)
    
    name = label_to_name.get(label, f'Class {label}')
    
    print(f"{name} (n={n_samples}):")
    print(f"  Mean within-class dissimilarity: {mean_dist:.3f}")
    print(f"  Range: [{min_dist:.3f}, {max_dist:.3f}]")
    
    # Warning for high within-class dissimilarity
    if mean_dist > 1.2:
        print(f"  ⚠️  WARNING: Very high! Samples may be misclassified or class is very diverse")
    elif mean_dist > 0.9:
        print(f"  ⚠️  High dissimilarity - check if predictions are correct")
    elif mean_dist < 0.6:
        print(f"  ✓ Low dissimilarity - class is consistent")

print("\n" + "=" * 80)
print("INTER-CLASS COMPARISON")
print("=" * 80)
print("\nChecking the most confused class pairs (low between-class dissimilarity)...\n")

# Compute average between-class dissimilarities
class_pairs = []
for i, label_i in enumerate(sorted(unique_labels)):
    for j, label_j in enumerate(sorted(unique_labels)):
        if i >= j:
            continue  # Skip diagonal and lower triangle
        
        indices_i = np.where(labels == label_i)[0]
        indices_j = np.where(labels == label_j)[0]
        
        # Get between-class dissimilarities
        between_rdm = rdm[np.ix_(indices_i, indices_j)]
        mean_between = np.mean(between_rdm)
        
        name_i = label_to_name.get(label_i, f'Class {label_i}')
        name_j = label_to_name.get(label_j, f'Class {label_j}')
        
        class_pairs.append((mean_between, name_i, name_j))

# Sort by dissimilarity (ascending - most similar first)
class_pairs.sort()

print("Most similar class pairs (potential confusion):")
for dist, name_i, name_j in class_pairs[:10]:
    print(f"  {dist:.3f}: {name_i} ↔ {name_j}")

print("\nMost dissimilar class pairs (well separated):")
for dist, name_i, name_j in class_pairs[-5:]:
    print(f"  {dist:.3f}: {name_i} ↔ {name_j}")
