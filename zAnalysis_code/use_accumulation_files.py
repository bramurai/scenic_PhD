#!/usr/bin/env python3
"""Utility to combine and recompute averaged activations from .accumulation files.

This demonstrates that you can:
1. Load activation sums directly from .accumulation/*.npy files
2. Combine classes in any way you want (e.g., group related classes)
3. Recompute averages with custom class groupings
4. Save new outputs without re-running the full extraction

Examples:
  - Combine all "Music" classes into one super-class
  - Group by broader AudioSet categories
  - Create per-domain averages (music vs speech vs environmental sounds)
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# Example: Load the original checkpoint to get class metadata
checkpoint_path = 'audioset_analysis_AV/checkpoint.pkl'
accumulation_dir = 'audioset_analysis_AV/.accumulation'
labels_csv = 'Video_csvs/audioset_labels.csv'

def load_class_metadata():
    """Load AudioSet class information."""
    labels_df = pd.read_csv(labels_csv)
    index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
    return labels_df, index_to_name

def load_activation_sum(class_idx: int, activation_name: str) -> np.ndarray:
    """Load a single activation sum from disk.
    
    This is what each .accumulation/*.npy file contains:
    - SUM of activations for all samples in that class
    - Not the average (no division by count)
    - Shape: (num_embeddings, embedding_dim) - e.g., (401, 768)
    """
    path = os.path.join(accumulation_dir, f'class_{class_idx}_{activation_name}.npy')
    if os.path.exists(path):
        return np.load(path)
    else:
        raise FileNotFoundError(f'No accumulation file for class {class_idx}, {activation_name}')

def get_sample_count(class_idx: int) -> int:
    """Get how many samples contributed to a class's activation sum."""
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    counts = checkpoint['counts']
    return counts.get(class_idx, 0)

def combine_classes(class_indices: List[int], 
                   activation_names: List[str],
                   counts_for_averaging: Dict[int, int]) -> Dict[str, np.ndarray]:
    """Combine multiple classes into a single averaged activation.
    
    Args:
        class_indices: List of AudioSet class indices to combine
        activation_names: List of activation layer names
        counts_for_averaging: Dict mapping class_idx -> number of samples
        
    Returns:
        Dict mapping activation_name -> averaged activation array
    """
    combined = {}
    
    for act_name in activation_names:
        # Load sums for all classes
        total_sum = None
        total_count = 0
        
        for class_idx in class_indices:
            try:
                act_sum = load_activation_sum(class_idx, act_name)
                count = counts_for_averaging[class_idx]
                
                if total_sum is None:
                    total_sum = act_sum.copy()
                else:
                    total_sum += act_sum
                
                total_count += count
            except FileNotFoundError:
                print(f"  Warning: Missing {act_name} for class {class_idx}")
                continue
        
        if total_sum is not None and total_count > 0:
            combined[act_name] = total_sum / total_count
        else:
            print(f"  Warning: Could not combine {act_name}")
    
    return combined

def demonstrate_usage():
    """Show how to use the .accumulation files for custom grouping."""
    print("=" * 80)
    print("CHECKPOINT & ACCUMULATION FILE STRUCTURE")
    print("=" * 80)
    
    # Load checkpoint
    import pickle
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    counts = checkpoint['counts']
    activation_names = checkpoint['activation_names']
    
    print(f"\n✓ Checkpoint contains:")
    print(f"  - processed_count: {checkpoint['processed_count']} samples processed")
    print(f"  - num_classes: {checkpoint['num_classes']} total AudioSet classes")
    print(f"  - counts: {len(counts)} classes have accumulated samples")
    print(f"  - activation_names: {len(activation_names)} layer types")
    
    print(f"\n✓ .accumulation directory contains:")
    print(f"  - {len(counts) * len(activation_names)} .npy files")
    print(f"  - Each file: class_IDX_LAYER_NAME.npy")
    print(f"  - Each file contains: SUM of activations (not averaged)")
    
    # Load class metadata
    labels_df, index_to_name = load_class_metadata()
    
    print(f"\n" + "=" * 80)
    print("EXAMPLE 1: Custom class grouping")
    print("=" * 80)
    
    # Find music-related classes
    music_classes = []
    for idx, name in index_to_name.items():
        if 'music' in name.lower():
            music_classes.append(idx)
    
    print(f"\nFound {len(music_classes)} music-related classes:")
    for class_idx in music_classes[:5]:
        count = counts.get(class_idx, 0)
        print(f"  class_{class_idx}: {index_to_name[class_idx]} ({count} samples)")
    if len(music_classes) > 5:
        print(f"  ... and {len(music_classes) - 5} more")
    
    # Combine them
    if music_classes:
        print(f"\nCombining all {len(music_classes)} music classes...")
        music_counts = {c: counts.get(c, 0) for c in music_classes}
        combined_music = combine_classes(
            music_classes, 
            activation_names,
            music_counts
        )
        print(f"✓ Created combined 'Music' activation with {len(combined_music)} layers")
        
        # Show example
        for layer_name in sorted(combined_music.keys())[:3]:
            arr = combined_music[layer_name]
            print(f"  {layer_name}: shape {arr.shape}, dtype {arr.dtype}")
    
    print(f"\n" + "=" * 80)
    print("EXAMPLE 2: Computing RDM for custom class combination")
    print("=" * 80)
    
    # Find speech classes
    speech_classes = []
    for idx, name in index_to_name.items():
        if 'speech' in name.lower():
            speech_classes.append(idx)
    
    print(f"\nFound {len(speech_classes)} speech-related classes:")
    for class_idx in speech_classes[:5]:
        count = counts.get(class_idx, 0)
        print(f"  class_{class_idx}: {index_to_name[class_idx]} ({count} samples)")
    
    # You could now:
    # 1. Compute RDM between music and speech groups
    # 2. Create RDM for any custom class combination
    # 3. Use any distance metric (correlation, euclidean, etc.)
    
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"""
✓ You ONLY need the .accumulation/ folder (not checkpoint.pkl)
  - Checkpoint just stores metadata for resuming extraction
  - Actual data is in the .npy files

✓ Each .npy file contains:
  - SUM of activations for a class (not averaged yet)
  - To get average: sum / num_samples_in_class

✓ You can combine classes by:
  1. Loading sums from multiple class .npy files
  2. Adding them together
  3. Dividing by total sample count

✓ You can create RDMs from any custom grouping:
  - Combine classes into super-classes
  - Group by AudioSet taxonomy
  - Create domain-specific groupings (music vs speech vs other)
  - Compute pairwise distances between combined groups

✓ Total storage needed:
  - All 527 classes × 24 layers = 12,648 .npy files (~65 GB)
  - No need to reload from raw audio or re-run model
    """)

def example_create_custom_rdm():
    """Example: Create RDM from custom class groupings."""
    import pickle
    from scipy.spatial.distance import correlation
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Create RDM with custom class groupings")
    print("=" * 80)
    
    # Load metadata
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    counts = checkpoint['counts']
    activation_names = checkpoint['activation_names']
    
    labels_df, index_to_name = load_class_metadata()
    
    # Define custom groupings
    groupings = {
        'music': [idx for idx, name in index_to_name.items() if 'music' in name.lower()],
        'speech': [idx for idx, name in index_to_name.items() if 'speech' in name.lower()],
        'wind': [idx for idx, name in index_to_name.items() if 'wind' in name.lower()],
        'dog': [idx for idx, name in index_to_name.items() if 'dog' in name.lower()],
        'cat': [idx for idx, name in index_to_name.items() if 'cat' in name.lower()],
    }
    
    print("\nDefined custom groupings:")
    for group_name, class_list in groupings.items():
        print(f"  {group_name}: {len(class_list)} classes")
    
    # Combine activations for each group
    group_activations = {}
    for group_name, class_list in groupings.items():
        if not class_list:
            continue
        class_counts = {c: counts.get(c, 0) for c in class_list}
        group_activations[group_name] = combine_classes(
            class_list,
            [activation_names[0]],  # Just use first layer for demo
            class_counts
        )
    
    # Compute RDM (pairwise correlation distance)
    print("\nComputing RDM from custom groups...")
    
    # Flatten activations for correlation
    group_vectors = {}
    for group_name, acts in group_activations.items():
        # Concatenate all layers and flatten
        vectors = []
        for layer_name in sorted(acts.keys()):
            vectors.append(acts[layer_name].flatten())
        group_vectors[group_name] = np.concatenate(vectors)
    
    # Compute pairwise correlation distances
    group_names = sorted(group_vectors.keys())
    n_groups = len(group_names)
    rdm = np.zeros((n_groups, n_groups))
    
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names):
            if i <= j:
                # Correlation distance
                dist = correlation(group_vectors[group1], group_vectors[group2])
                rdm[i, j] = dist
                rdm[j, i] = dist
    
    print(f"\nRDM shape: {rdm.shape}")
    print(f"Groups: {group_names}")
    print("\nRDM matrix (correlation distances):")
    print(rdm)
    
    print("\nInterpretation:")
    print("  - 0 = identical activations")
    print("  - 1 = completely opposite")
    print("  - 0.5 = orthogonal")

if __name__ == '__main__':
    demonstrate_usage()
    example_create_custom_rdm()
