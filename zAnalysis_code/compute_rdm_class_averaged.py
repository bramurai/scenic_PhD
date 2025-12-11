#!/usr/bin/env python3
"""Compute RDMs from class-averaged MBT activations.

This script works with the output from extract_mbt_activations_class_averaged.py.

Usage:
  python compute_rdm_class_averaged.py \
    --activation_file=audioset_analysis_12-9-2025/class_averaged_activations.npz \
    --output_dir=RDM_from_averaged_minimal \
    --distance_metric=correlation \
    --audioset_labels_csv=Video_csvs/audioset_labels.csv
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
from absl import app, flags, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

FLAGS = flags.FLAGS

flags.DEFINE_string('activation_file', None, 'Path to class_averaged_activations.npz')
flags.DEFINE_string('output_dir', 'RDM_class_averaged', 'Output directory for RDMs')
flags.DEFINE_enum('distance_metric', 'correlation', 
                  ['correlation', 'euclidean', 'cosine', 'cityblock'],
                  'Distance metric for RDM computation')
flags.DEFINE_list('layers', None, 
                  'Specific layers to analyze (comma-separated, e.g., "0,5,11"). '
                  'If None, analyzes all layers.')
flags.DEFINE_bool('standardize', True, 
                  'Standardize activations before computing distances')
flags.DEFINE_bool('plot_dendrograms', False, 
                  'Include hierarchical clustering dendrograms')
flags.DEFINE_string('audioset_labels_csv', None,
                    'Path to audioset_labels.csv for class names')
flags.DEFINE_integer('min_samples_per_class', 1,
                     'Minimum number of samples required per class to include in RDM')

flags.mark_flag_as_required('activation_file')
flags.mark_flag_as_required('audioset_labels_csv')


def load_class_averaged_activations(activation_file: str) -> Tuple[Dict, np.ndarray, List[str]]:
    """Load class-averaged activations from npz file.
    
    Returns:
        class_activations: Dict mapping (class_idx, layer_name) -> activation array
        class_indices: Array of class indices that have samples
        layer_names: List of unique layer names
    """
    logging.info(f'Loading class-averaged activations from {activation_file}')
    
    data = np.load(activation_file, allow_pickle=True)
    
    # Extract metadata
    class_names = data['class_names']
    samples_per_class = data['samples_per_class']
    num_classes = int(data['num_classes'])
    
    logging.info(f'File contains data for {num_classes} classes')
    logging.info(f'Total samples processed: {data["num_samples_processed"]}')
    
    # Find which classes have samples
    class_indices = np.where(samples_per_class >= FLAGS.min_samples_per_class)[0]
    logging.info(f'Classes with >= {FLAGS.min_samples_per_class} samples: {len(class_indices)}')
    
    # Parse activation keys to find layer names
    layer_names_set = set()
    class_activations = {}
    
    for key in data.keys():
        if key.startswith('class_'):
            # Parse key: class_72_encoder_block_L0_rgb_output
            parts = key.split('_', 2)  # Split into ['class', '72', 'encoder_block_L0_rgb_output']
            if len(parts) == 3:
                class_idx = int(parts[1])
                layer_name = parts[2]
                
                # Only include classes with enough samples
                if class_idx in class_indices:
                    layer_names_set.add(layer_name)
                    class_activations[(class_idx, layer_name)] = data[key]
    
    layer_names = sorted(layer_names_set, key=lambda x: (
        int(x.split('_L')[1].split('_')[0]) if '_L' in x else 0,
        'audio' in x  # Sort audio after rgb within same layer
    ))
    
    logging.info(f'Found {len(layer_names)} unique layers:')
    for layer in layer_names:
        logging.info(f'  - {layer}')
    
    return class_activations, class_indices, layer_names, samples_per_class


def load_audioset_labels(csv_path: str) -> Dict[int, str]:
    """Load AudioSet label mapping from CSV.
    
    Returns:
        index_to_name: Dict mapping class index to display name
    """
    logging.info(f'Loading AudioSet labels from {csv_path}')
    
    labels_df = pd.read_csv(csv_path)
    index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
    
    logging.info(f'Loaded {len(index_to_name)} label mappings')
    return index_to_name


def extract_layer_activations_for_classes(class_activations: Dict,
                                          class_indices: np.ndarray,
                                          layer_name: str) -> np.ndarray:
    """Extract activations for a specific layer across all classes.
    
    Returns:
        layer_acts: Array of shape (num_classes, feature_dim)
    """
    acts_list = []
    
    for class_idx in class_indices:
        key = (class_idx, layer_name)
        if key not in class_activations:
            raise ValueError(f'Missing activation for class {class_idx}, layer {layer_name}')
        
        act = class_activations[key]
        # Flatten to 1D
        act_flat = act.flatten()
        acts_list.append(act_flat)
    
    return np.array(acts_list)  # (num_classes, feature_dim)


def compute_rdm(activations: np.ndarray, 
                metric: str = 'correlation',
                standardize: bool = True) -> np.ndarray:
    """Compute Representational Dissimilarity Matrix.
    
    Args:
        activations: Array of shape (num_classes, feature_dim)
        metric: Distance metric
        standardize: Whether to standardize features
    
    Returns:
        rdm: Dissimilarity matrix of shape (num_classes, num_classes)
    """
    if standardize:
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)
    
    distances = pdist(activations, metric=metric)
    rdm = squareform(distances)
    
    return rdm


def plot_rdm(rdm: np.ndarray,
             class_indices: np.ndarray,
             samples_per_class: np.ndarray,
             index_to_name: Dict[int, str],
             layer_name: str,
             output_path: str,
             plot_dendrogram: bool = True):
    """Plot class-level RDM with optional hierarchical clustering."""
    
    num_classes = len(class_indices)
    
    # Create label names
    class_labels = []
    for idx in class_indices:
        name = index_to_name.get(idx, f'Class {idx}')
        count = samples_per_class[idx]
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + '...'
        class_labels.append(f'{name} (n={count})')
    
    # Create figure
    if plot_dendrogram:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                             hspace=0.05, wspace=0.05)
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        ax_rdm = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_rdm = plt.subplots(figsize=(16, 14))
    
    # Compute hierarchical clustering
    linkage_matrix = linkage(squareform(rdm), method='average')
    
    # Plot dendrograms
    if plot_dendrogram:
        dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True)
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        for spine in ax_dendro_top.spines.values():
            spine.set_visible(False)
        
        dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left, 
                                orientation='left', no_labels=True)
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        for spine in ax_dendro_left.spines.values():
            spine.set_visible(False)
        
        order = dendro_top['leaves']
    else:
        order = np.arange(num_classes)
    
    # Reorder RDM
    rdm_sorted = rdm[order][:, order]
    labels_sorted = [class_labels[i] for i in order]
    
    # Plot RDM
    im = ax_rdm.imshow(rdm_sorted, cmap='viridis', aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_rdm, fraction=0.046, pad=0.04)
    cbar.set_label('Dissimilarity', rotation=270, labelpad=20, fontsize=12)
    
    # Labels
    ax_rdm.set_xticks(range(num_classes))
    ax_rdm.set_xticklabels(labels_sorted, rotation=90, ha='center', fontsize=8, va='top')
    ax_rdm.set_yticks(range(num_classes))
    ax_rdm.set_yticklabels(labels_sorted, fontsize=8)
    
    # Title
    ax_rdm.set_title(f'Class-Level RDM: {layer_name}\n({FLAGS.distance_metric} distance)', 
                     fontsize=14, pad=15)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f'  Saved RDM plot to {output_path}')


def plot_dissimilarity_evolution(rdm_results: Dict[str, np.ndarray],
                                 class_indices: np.ndarray,
                                 index_to_name: Dict[int, str],
                                 output_dir: str):
    """Plot how dissimilarity evolves across layers."""
    
    logging.info('\nPlotting dissimilarity evolution across layers...')
    
    layer_names = sorted(rdm_results.keys(), 
                        key=lambda x: int(x.split('_L')[1].split('_')[0]) if '_L' in x else 0)
    
    if len(layer_names) < 2:
        logging.info('  Need at least 2 layers. Skipping.')
        return
    
    num_classes = len(class_indices)
    
    # Compute statistics
    mean_dissim = []
    std_dissim = []
    max_dissim = []
    min_dissim = []
    
    for layer_name in layer_names:
        rdm = rdm_results[layer_name]
        # Get off-diagonal elements (between-class dissimilarities)
        mask = ~np.eye(num_classes, dtype=bool)
        off_diag = rdm[mask]
        
        mean_dissim.append(np.mean(off_diag))
        std_dissim.append(np.std(off_diag))
        max_dissim.append(np.max(off_diag))
        min_dissim.append(np.min(off_diag))
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(layer_names))
    
    # Clean layer names
    layer_labels = []
    for name in layer_names:
        if 'encoder_block_L' in name:
            parts = name.split('_')
            layer_num = parts[2][1:]
            modality = 'Audio' if 'audio' in name else 'RGB'
            layer_labels.append(f'L{layer_num} {modality}')
        else:
            layer_labels.append(name)
    
    # Plot mean with error bars
    ax.errorbar(x_pos, mean_dissim, yerr=std_dissim, 
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                label='Mean ± Std', color='blue')
    
    # Plot range
    ax.fill_between(x_pos, min_dissim, max_dissim, alpha=0.2, 
                    color='blue', label='Min-Max range')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Between-Class Dissimilarity', fontsize=12)
    ax.set_title('Dissimilarity Evolution Across Layers', fontsize=14, pad=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    evolution_path = os.path.join(output_dir, 'dissimilarity_evolution.png')
    plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f'  Saved evolution plot to {evolution_path}')
    
    # Also plot class-pair heatmap if not too many classes
    if num_classes <= 40:
        fig, ax = plt.subplots(figsize=(max(14, len(layer_names) * 0.8), 
                                       max(10, num_classes * 0.6)))
        
        # Create matrix: rows = class pairs, columns = layers
        class_pairs = []
        class_pair_labels = []
        
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                class_pairs.append((i, j))
                name_i = index_to_name.get(class_indices[i], f'C{class_indices[i]}')
                name_j = index_to_name.get(class_indices[j], f'C{class_indices[j]}')
                if len(name_i) > 15:
                    name_i = name_i[:12] + '...'
                if len(name_j) > 15:
                    name_j = name_j[:12] + '...'
                class_pair_labels.append(f'{name_i} - {name_j}')
        
        # Fill dissimilarity matrix
        dissim_matrix = np.zeros((len(class_pairs), len(layer_names)))
        
        for layer_idx, layer_name in enumerate(layer_names):
            rdm = rdm_results[layer_name]
            for pair_idx, (i, j) in enumerate(class_pairs):
                dissim_matrix[pair_idx, layer_idx] = rdm[i, j]
        
        im = ax.imshow(dissim_matrix, cmap='viridis', aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Dissimilarity', rotation=270, labelpad=20)
        
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(class_pairs)))
        ax.set_yticklabels(class_pair_labels, fontsize=7)
        
        ax.set_title('Class-Pair Dissimilarity Evolution', fontsize=13, pad=10)
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'class_pair_evolution_heatmap.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f'  Saved class-pair heatmap to {heatmap_path}')


def main(argv):
    del argv
    
    logging.info('='*80)
    logging.info('RDM Computation from Class-Averaged Activations')
    logging.info('='*80)
    
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Load data
    logging.info('\n[1/4] Loading class-averaged activations...')
    class_activations, class_indices, layer_names, samples_per_class = \
        load_class_averaged_activations(FLAGS.activation_file)
    
    num_classes = len(class_indices)
    logging.info(f'Processing {num_classes} classes')
    
    # Load labels
    logging.info('\n[2/4] Loading AudioSet labels...')
    index_to_name = load_audioset_labels(FLAGS.audioset_labels_csv)
    
    # Print class info
    logging.info(f'\nClasses included in RDM:')
    for idx in class_indices[:10]:  # Show first 10
        name = index_to_name.get(idx, f'Class {idx}')
        count = samples_per_class[idx]
        logging.info(f'  {idx}: {name} ({count} samples)')
    if num_classes > 10:
        logging.info(f'  ... and {num_classes - 10} more classes')
    
    # Determine layers to analyze
    if FLAGS.layers:
        layers_to_analyze = []
        requested_layers = [int(x) for x in FLAGS.layers]
        for layer_num in requested_layers:
            # Find both RGB and audio for this layer
            for layer_name in layer_names:
                if f'_L{layer_num}_' in layer_name:
                    layers_to_analyze.append(layer_name)
    else:
        layers_to_analyze = layer_names
    
    logging.info(f'\n[3/4] Computing RDMs for {len(layers_to_analyze)} layers...')
    
    rdm_results = {}
    
    for layer_name in layers_to_analyze:
        logging.info(f'\nProcessing layer: {layer_name}')
        
        # Extract activations
        layer_acts = extract_layer_activations_for_classes(
            class_activations, class_indices, layer_name)
        logging.info(f'  Activation shape: {layer_acts.shape}')
        
        # Compute RDM
        rdm = compute_rdm(layer_acts, 
                         metric=FLAGS.distance_metric,
                         standardize=FLAGS.standardize)
        logging.info(f'  RDM shape: {rdm.shape}')
        logging.info(f'  Mean dissimilarity: {np.mean(rdm):.4f}')
        logging.info(f'  Std dissimilarity: {np.std(rdm):.4f}')
        
        rdm_results[layer_name] = rdm
        
        # Save RDM
        rdm_path = os.path.join(FLAGS.output_dir, f'rdm_{layer_name}.npz')
        np.savez_compressed(rdm_path,
                           rdm=rdm,
                           class_indices=class_indices,
                           samples_per_class=samples_per_class[class_indices],
                           layer_name=layer_name,
                           metric=FLAGS.distance_metric)
        logging.info(f'  Saved to {rdm_path}')
    
    # Plot RDMs
    logging.info(f'\n[4/4] Plotting RDMs...')
    
    for layer_name, rdm in rdm_results.items():
        logging.info(f'\nPlotting {layer_name}...')
        plot_path = os.path.join(FLAGS.output_dir, f'rdm_{layer_name}.png')
        plot_rdm(rdm, class_indices, samples_per_class, index_to_name,
                layer_name, plot_path, FLAGS.plot_dendrograms)
    
    # Plot evolution
    if len(rdm_results) > 1:
        plot_dissimilarity_evolution(rdm_results, class_indices, 
                                     index_to_name, FLAGS.output_dir)
    
    # Save summary
    summary = {
        'num_classes': num_classes,
        'class_indices': class_indices.tolist(),
        'samples_per_class': {int(idx): int(samples_per_class[idx]) 
                             for idx in class_indices},
        'layers_analyzed': layers_to_analyze,
        'distance_metric': FLAGS.distance_metric,
        'standardized': FLAGS.standardize,
        'class_names': {int(idx): index_to_name.get(idx, f'Class {idx}') 
                       for idx in class_indices}
    }
    
    summary_path = os.path.join(FLAGS.output_dir, 'summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    logging.info('\n' + '='*80)
    logging.info('RDM Computation Complete!')
    logging.info(f'Analyzed {num_classes} classes across {len(layers_to_analyze)} layers')
    logging.info(f'Outputs saved to: {FLAGS.output_dir}')
    logging.info('='*80)


if __name__ == '__main__':
    app.run(main)
