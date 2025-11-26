#!/usr/bin/env python3
"""Quick reference: How to combine classes using .accumulation files only.

This shows the minimal code needed to:
1. Load activation sums from .accumulation/
2. Combine multiple classes
3. Compute averaged activations
4. Create custom RDMs
"""

import numpy as np
import os
from pathlib import Path

# Paths
ACCUM_DIR = Path('audioset_analysis_AV/.accumulation')

def load_class_activation_sum(class_idx: int, layer_name: str) -> np.ndarray:
    """Load the SUM of activations for a class at a layer."""
    path = ACCUM_DIR / f'class_{class_idx}_{layer_name}.npy'
    return np.load(path)

def load_all_class_sums(class_indices: list, layer_name: str) -> dict:
    """Load activation sums for multiple classes at a single layer."""
    return {
        class_idx: load_class_activation_sum(class_idx, layer_name)
        for class_idx in class_indices
    }

# ============================================================================
# EXAMPLE 1: Combine two classes
# ============================================================================

print("=" * 80)
print("EXAMPLE 1: Combine two classes")
print("=" * 80)

# Say we want to combine:
# - class_137 (Music): 45 samples
# - class_138 (Speech): 89 samples

class_137_sum = load_class_activation_sum(137, 'encoder_block_L0_rgb_output')
class_138_sum = load_class_activation_sum(138, 'encoder_block_L0_rgb_output')

print(f"class_137 shape: {class_137_sum.shape}")  # (401, 768)
print(f"class_138 shape: {class_138_sum.shape}")  # (401, 768)

# Combine: add the sums
combined_sum = class_137_sum + class_138_sum
print(f"combined shape: {combined_sum.shape}")

# Average: divide by total sample count
total_samples = 45 + 89  # 134
combined_avg = combined_sum / total_samples
print(f"combined average: {combined_avg.shape}")
print(f"  → This is the average activation for the 'Music+Speech' group")

# ============================================================================
# EXAMPLE 2: Combine many classes (Music super-class)
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Combine many classes into a super-class")
print("=" * 80)

# Imagine these are all music-related classes
music_class_indices = [137, 138, 140, 141, 143, 145]
music_class_samples = {
    137: 45,   # Music - Classical
    138: 89,   # Music - Electronic
    140: 23,   # Music - Jazz
    141: 56,   # Music - Pop
    143: 34,   # Music - Rock
    145: 12,   # Music - Blues
}

layer_name = 'encoder_block_L0_rgb_output'

# Load all sums
music_sums = load_all_class_sums(music_class_indices, layer_name)

# Combine by summing
music_combined_sum = np.zeros_like(list(music_sums.values())[0])
for class_idx, activation_sum in music_sums.items():
    music_combined_sum += activation_sum

# Average by total count
total_music_samples = sum(music_class_samples.values())
music_avg = music_combined_sum / total_music_samples

print(f"Combined {len(music_class_indices)} music classes")
print(f"Total samples: {total_music_samples}")
print(f"Result shape: {music_avg.shape}")

# ============================================================================
# EXAMPLE 3: Vectorize for correlation distance
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Compute correlation distance between groups")
print("=" * 80)

from scipy.spatial.distance import correlation

# Create two groups
group_a_classes = [137, 140, 141]  # Music subgenres
group_b_classes = [200, 201, 202]  # Speech categories

# Get all layer names
all_files = list(ACCUM_DIR.glob('*.npy'))
layer_names = list(set(
    f.name.replace('class_', '').split('_', 1)[1].replace('.npy', '')
    for f in all_files
))[:3]  # Just first 3 for demo

print(f"Using {len(layer_names)} layers for correlation")

# Combine and vectorize each group
def get_group_vector(class_indices, layer_names):
    vectors = []
    for layer_name in layer_names:
        sums = load_all_class_sums(class_indices, layer_name)
        group_sum = np.zeros_like(list(sums.values())[0])
        for s in sums.values():
            group_sum += s
        vectors.append(group_sum.flatten())
    return np.concatenate(vectors)

vec_a = get_group_vector(group_a_classes, layer_names)
vec_b = get_group_vector(group_b_classes, layer_names)

print(f"Group A vector shape: {vec_a.shape}")
print(f"Group B vector shape: {vec_b.shape}")

# Compute correlation distance
dist = correlation(vec_a, vec_b)
print(f"Correlation distance: {dist:.4f}")
print(f"  → 0.0 = identical, 1.0 = opposite, 0.5 = orthogonal")

# ============================================================================
# EXAMPLE 4: Create 2D RDM for custom classes
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Create RDM for custom class groups")
print("=" * 80)

# Define groups
groups = {
    'Music': [137, 140, 141],
    'Speech': [200, 201, 202],
    'Nature': [300, 301, 302],
}

# Get vectors for each group
group_vectors = {}
for group_name, class_indices in groups.items():
    group_vectors[group_name] = get_group_vector(class_indices, layer_names)
    print(f"{group_name}: vector shape {group_vectors[group_name].shape}")

# Compute pairwise distances
group_names = sorted(group_vectors.keys())
n = len(group_names)
rdm = np.zeros((n, n))

for i, name1 in enumerate(group_names):
    for j, name2 in enumerate(group_names):
        if i != j:
            rdm[i, j] = correlation(group_vectors[name1], group_vectors[name2])

print(f"\nRDM shape: {rdm.shape}")
print(f"RDM:\n{rdm}")
print(f"\nGroup order: {group_names}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY - Key Operations")
print("=" * 80)

print("""
1. Load single class sum:
   >>> sum_arr = np.load('.accumulation/class_137_encoder_block_L0_rgb_output.npy')

2. Combine multiple classes:
   >>> combined_sum = sum_arr1 + sum_arr2 + sum_arr3
   >>> combined_avg = combined_sum / total_samples

3. Vectorize for distances:
   >>> vector = np.concatenate([arr.flatten() for arr in [arr1, arr2, ...]])

4. Compute correlation:
   >>> from scipy.spatial.distance import correlation
   >>> dist = correlation(vec1, vec2)

5. Create RDM matrix:
   >>> rdm[i, j] = correlation(group_vectors[name1], group_vectors[name2])

All operations work directly on .accumulation/ files!
No need for checkpoint.pkl or original data.
""")
