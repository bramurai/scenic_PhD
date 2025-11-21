#!/usr/bin/env python3
"""Compute Representational Dissimilarity Matrices (RDMs) from MBT activations.

This script:
1. Loads extracted activations from extract_mbt_activations.py
2. Computes pairwise distances between samples for each layer
3. Creates RDMs organized by labels
4. Visualizes RDMs with hierarchical clustering

Usage:
  python compute_rdm.py \
    --activation_dir=activation_analysis \
    --output_dir=rdm_analysis \
    --distance_metric=correlation \
    --layers=encoder_block_L11_rgb_output,encoder_block_L11_audio_output
  
  On first run, loads all sample_*.npz files and saves activations_cache.pkl.
  Subsequent runs load from cache (much faster). Use --use_cache=False to disable.
"""

import os
import glob
import pickle
from typing import Dict, List, Optional, Tuple
from absl import app, flags, logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

FLAGS = flags.FLAGS

flags.DEFINE_string('activation_dir', None, 'Directory with extracted activations')
flags.DEFINE_string('output_dir', 'rdm_analysis', 'Output directory for RDMs')
flags.DEFINE_enum('distance_metric', 'correlation', 
                  ['correlation', 'euclidean', 'cosine', 'cityblock'],
                  'Distance metric for RDM computation')
flags.DEFINE_list('layers', None, 
                  'Specific layers to analyze (comma-separated). '
                  'If None, analyzes all encoder outputs.')
flags.DEFINE_bool('standardize', True, 
                  'Standardize activations before computing distances')
flags.DEFINE_bool('plot_dendrograms', True, 
                  'Include hierarchical clustering dendrograms')
flags.DEFINE_string('label_mapping_file', None,
                    'Path to label_mapping.txt file. If None, looks in activation_dir parent.')
flags.DEFINE_bool('use_cache', True,
                  'Use cached activations if available (saves loading time)')
flags.DEFINE_bool('average_by_class', False,
                  'Average activations within each class before computing RDM. '
                  'Results in class-level RDM instead of sample-level RDM.')

flags.mark_flag_as_required('activation_dir')


def load_all_activations(activation_dir: str) -> Tuple[List[Dict], List[int], List[str]]:
    """Load all activation files.
    
    Returns:
        activations: List of activation dicts (one per sample)
        sample_indices: Sample indices
        available_layers: List of all layer names found
    """
    # Check for cached version
    cache_path = os.path.join(activation_dir, 'activations_cache.pkl')
    
    if FLAGS.use_cache and os.path.exists(cache_path):
        logging.info(f'Loading activations from cache: {cache_path}')
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            logging.info(f'  Loaded {len(cache_data["activations"])} samples from cache')
            logging.info(f'  Available layers: {len(cache_data["available_layers"])}')
            return cache_data['activations'], cache_data['sample_indices'], cache_data['available_layers']
        except Exception as e:
            logging.warning(f'Failed to load cache: {e}. Loading from individual files...')
    
    logging.info(f'Loading activations from {activation_dir}...')
    
    # Find all sample files
    sample_files = sorted(glob.glob(os.path.join(activation_dir, 'sample_*.npz')))
    
    if not sample_files:
        raise ValueError(f'No sample files found in {activation_dir}')
    
    logging.info(f'Found {len(sample_files)} samples')
    
    # Load all samples
    activations = []
    sample_indices = []
    available_layers = None
    
    for i, sample_file in enumerate(sample_files):
        if (i + 1) % 10 == 0 or i == 0:
            logging.info(f'Loading sample file {i+1}/{len(sample_files)}...')
        data = np.load(sample_file)
        activations.append({key: data[key] for key in data.keys()})
        sample_indices.append(int(data['sample_idx']))
        
        # Get layer names from first sample
        if available_layers is None:
            available_layers = [key for key in data.keys() 
                              if key not in ['logits', 'sample_idx']]
    
    logging.info(f'Available layers: {len(available_layers)}')
    for layer in available_layers:
        logging.info(f'  - {layer}')
    
    # Save cache for next time
    if FLAGS.use_cache:
        logging.info(f'Saving cache to {cache_path}...')
        cache_data = {
            'activations': activations,
            'sample_indices': sample_indices,
            'available_layers': available_layers
        }
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f'  Cache saved successfully')
        except Exception as e:
            logging.warning(f'Failed to save cache: {e}')
    
    return activations, sample_indices, available_layers


def load_label_mapping(activation_dir: str, label_mapping_file: Optional[str] = None) -> Dict[int, str]:
    """Load label mapping from file.
    
    Returns:
        label_to_name: Dict mapping label index to label name
    """
    # Try to find label_mapping.txt
    if label_mapping_file and os.path.exists(label_mapping_file):
        mapping_path = label_mapping_file
    else:
        # Look in activation_dir or parent directory
        candidates = [
            os.path.join(activation_dir, 'label_mapping.txt'),
            os.path.join(os.path.dirname(activation_dir), 'label_mapping.txt'),
        ]
        
        mapping_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                mapping_path = candidate
                break
    
    if mapping_path is None:
        logging.warning('No label_mapping.txt found, using numeric labels')
        return {}
    
    logging.info(f'Loading label mapping from {mapping_path}')
    
    label_to_name = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    idx, name = parts
                    label_to_name[int(idx)] = name
    
    logging.info(f'Loaded {len(label_to_name)} label mappings')
    return label_to_name


def get_labels_from_logits(activations: List[Dict]) -> np.ndarray:
    """Extract labels from activation data.
    
    First tries to get ground truth labels from 'label' field.
    Falls back to predicted labels (argmax of logits) if not available.
    
    Returns:
        labels: Array of shape (num_samples,) with class indices
    """
    # Check if we have ground truth labels
    if 'label' in activations[0]:
        logging.info('Using ground truth labels from data')
        labels_list = []
        for sample in activations:
            label = sample['label']
            # Remove batch dimension if present
            if label.ndim > 1 and label.shape[0] == 1:
                label = label[0]
            # Convert one-hot to index if needed
            if label.ndim > 0 and label.shape[0] > 1:
                label = np.argmax(label)
            labels_list.append(int(label))
        return np.array(labels_list)
    else:
        # Fall back to predicted labels
        logging.warning('No ground truth labels found, using predicted labels (argmax of logits)')
        logits_list = []
        for sample in activations:
            logits = sample['logits']
            # Remove batch dimension if present
            if logits.ndim > 1 and logits.shape[0] == 1:
                logits = logits[0]
            logits_list.append(logits)
        
        logits_array = np.array(logits_list)  # (num_samples, num_classes)
        labels = np.argmax(logits_array, axis=1)  # (num_samples,)
        
        return labels


def extract_layer_activations(activations: List[Dict], layer_name: str) -> np.ndarray:
    """Extract activations for a specific layer from all samples.
    
    Returns:
        layer_activations: Array of shape (num_samples, feature_dim)
    """
    layer_acts = []
    
    for sample in activations:
        if layer_name not in sample:
            raise ValueError(f'Layer {layer_name} not found in sample')
        
        act = sample[layer_name]
        
        # Flatten to 1D if needed (remove batch/sequence dimensions)
        # Typical shapes:
        #   - Encoder outputs: (1, seq_len, hidden_dim) or (seq_len, hidden_dim)
        #   - Bottleneck tokens: (1, 5, hidden_dim) or (5, hidden_dim)
        #   - Attention: (seq_len, seq_len) or (1, seq_len, seq_len)
        
        # Remove batch dimension if present
        while act.ndim > 0 and act.shape[0] == 1:
            act = act[0]
        
        # Flatten to 1D
        act_flat = act.flatten()
        layer_acts.append(act_flat)
    
    return np.array(layer_acts)  # (num_samples, feature_dim)


def average_activations_by_class(activations: np.ndarray, 
                                  labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Average activations within each class.
    
    Args:
        activations: Array of shape (num_samples, feature_dim)
        labels: Array of shape (num_samples,) with class indices
    
    Returns:
        class_activations: Array of shape (num_classes, feature_dim)
        class_labels: Array of shape (num_classes,) with class indices
        class_counts: List of sample counts per class
    """
    unique_labels = np.unique(labels)
    class_activations = []
    class_counts = []
    
    for label in unique_labels:
        # Get all samples for this class
        mask = labels == label
        class_samples = activations[mask]
        
        # Average across samples
        class_mean = np.mean(class_samples, axis=0)
        class_activations.append(class_mean)
        class_counts.append(int(mask.sum()))
    
    logging.info(f'Averaged {len(labels)} samples into {len(unique_labels)} classes')
    for i, (label, count) in enumerate(zip(unique_labels, class_counts)):
        logging.info(f'  Class {label}: {count} samples')
    
    return np.array(class_activations), unique_labels, class_counts


def compute_rdm(activations: np.ndarray, 
                metric: str = 'correlation',
                standardize: bool = True) -> np.ndarray:
    """Compute Representational Dissimilarity Matrix.
    
    Args:
        activations: Array of shape (num_samples, feature_dim)
        metric: Distance metric ('correlation', 'euclidean', 'cosine', 'cityblock')
        standardize: Whether to standardize features before computing distances
    
    Returns:
        rdm: Dissimilarity matrix of shape (num_samples, num_samples)
    """
    if standardize:
        scaler = StandardScaler()
        activations = scaler.fit_transform(activations)
    
    # Compute pairwise distances
    distances = pdist(activations, metric=metric)
    rdm = squareform(distances)
    
    return rdm


def plot_rdm(rdm: np.ndarray,
             labels: np.ndarray,
             label_to_name: Dict[int, str],
             layer_name: str,
             output_path: str,
             plot_dendrogram: bool = True):
    """Plot RDM with optional hierarchical clustering.
    
    Args:
        rdm: Dissimilarity matrix (num_samples, num_samples)
        labels: Class labels for each sample (num_samples,)
        label_to_name: Mapping from label index to name
        layer_name: Name of the layer
        output_path: Path to save figure
        plot_dendrogram: Whether to include dendrogram
    """
    num_samples = rdm.shape[0]
    
    # Create label names for display
    if label_to_name:
        label_names = [label_to_name.get(label, f'Class {label}') for label in labels]
    else:
        label_names = [f'Class {label}' for label in labels]
    
    # Compute hierarchical clustering
    linkage_matrix = linkage(squareform(rdm), method='average')
    
    # Create figure
    if plot_dendrogram:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4],
                             hspace=0.05, wspace=0.05)
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        ax_rdm = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_rdm = plt.subplots(figsize=(12, 10))
    
    # Plot dendrograms
    if plot_dendrogram:
        dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True)
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        ax_dendro_top.spines['top'].set_visible(False)
        ax_dendro_top.spines['right'].set_visible(False)
        ax_dendro_top.spines['bottom'].set_visible(False)
        ax_dendro_top.spines['left'].set_visible(False)
        
        dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left, 
                                orientation='left', no_labels=True)
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        ax_dendro_left.spines['top'].set_visible(False)
        ax_dendro_left.spines['right'].set_visible(False)
        ax_dendro_left.spines['bottom'].set_visible(False)
        ax_dendro_left.spines['left'].set_visible(False)
        
        # Reorder RDM based on clustering
        order = dendro_top['leaves']
    else:
        # Sort by label for better visualization
        order = np.argsort(labels)
    
    rdm_sorted = rdm[order][:, order]
    labels_sorted = labels[order]
    label_names_sorted = [label_names[i] for i in order]
    
    # Plot RDM
    im = ax_rdm.imshow(rdm_sorted, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_rdm, fraction=0.046, pad=0.04)
    cbar.set_label('Dissimilarity', rotation=270, labelpad=20)
    
    # Find class boundaries and midpoints for labeling
    unique_labels_sorted = []
    boundaries = []
    class_midpoints = []
    class_names = []
    class_counts = []
    
    for i, label in enumerate(labels_sorted):
        if not unique_labels_sorted or label != unique_labels_sorted[-1]:
            if unique_labels_sorted:
                # Mark end of previous class
                boundaries.append(i)
                # Calculate midpoint of previous class
                class_midpoints.append((boundaries[-2] + i) / 2)
                count = i - boundaries[-2]
                class_counts.append(count)
                base_name = label_to_name.get(unique_labels_sorted[-1], 
                                             f'Class {unique_labels_sorted[-1]}')
                class_names.append(f'{base_name} (n={count})')
            unique_labels_sorted.append(label)
            boundaries.append(i)
    
    # Add final class
    boundaries.append(num_samples)
    class_midpoints.append((boundaries[-2] + num_samples) / 2)
    count = num_samples - boundaries[-2]
    class_counts.append(count)
    base_name = label_to_name.get(unique_labels_sorted[-1], 
                                 f'Class {unique_labels_sorted[-1]}')
    class_names.append(f'{base_name} (n={count})')
    
    # Set tick positions at class midpoints
    ax_rdm.set_xticks(class_midpoints)
    ax_rdm.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax_rdm.set_yticks(class_midpoints)
    ax_rdm.set_yticklabels(class_names, fontsize=9)
    
    # Add grid lines at class boundaries
    for boundary in boundaries[1:-1]:  # Skip first (0) and last (num_samples)
        ax_rdm.axhline(y=boundary - 0.5, color='white', linewidth=1.5, alpha=0.7)
        ax_rdm.axvline(x=boundary - 0.5, color='white', linewidth=1.5, alpha=0.7)
    
    ax_rdm.set_title(f'RDM: {layer_name}\n({FLAGS.distance_metric} distance)', 
                     fontsize=12, pad=10)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f'  Saved RDM plot to {output_path}')


def compute_label_averaged_rdm(rdm: np.ndarray, 
                                labels: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Average RDM within each label category.
    
    This computes between-class dissimilarity by averaging pairwise distances
    between all samples from class i and all samples from class j.
    
    For within-class dissimilarity (diagonal, i==j), we have two options:
    1. Average pairwise distances within the class (mean distance between different samples)
    2. Set to 0 (representing that a class is identical to itself)
    
    We use option 2 to match the behavior of --average_by_class flag.
    
    Args:
        rdm: Dissimilarity matrix (num_samples, num_samples)
        labels: Class labels (num_samples,)
    
    Returns:
        averaged_rdm: Label-averaged RDM (num_labels, num_labels)
        unique_labels: List of unique label indices
    """
    unique_labels = sorted(np.unique(labels))
    num_labels = len(unique_labels)
    
    averaged_rdm = np.zeros((num_labels, num_labels))
    
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i == j:
                # Within-class: set to 0 (class is identical to itself)
                # This matches the behavior when using --average_by_class
                averaged_rdm[i, j] = 0.0
            else:
                # Between-class: average all pairwise distances
                mask_i = labels == label_i
                mask_j = labels == label_j
                
                # Extract sub-matrix (distances between samples from class i and class j)
                sub_rdm = rdm[np.ix_(mask_i, mask_j)]
                
                # Average all pairwise distances
                averaged_rdm[i, j] = np.mean(sub_rdm)
    
    return averaged_rdm, unique_labels


def plot_label_averaged_rdm(averaged_rdm: np.ndarray,
                            unique_labels: List[int],
                            label_to_name: Dict[int, str],
                            layer_name: str,
                            output_path: str):
    """Plot label-averaged RDM.
    
    Args:
        averaged_rdm: Label-averaged dissimilarity matrix (num_labels, num_labels)
        unique_labels: List of unique label indices
        label_to_name: Mapping from label index to name
        layer_name: Name of the layer
        output_path: Path to save figure
    """
    num_labels = len(unique_labels)
    
    # Create label names
    if label_to_name:
        label_names = [label_to_name.get(label, f'Class {label}') 
                      for label in unique_labels]
    else:
        label_names = [f'Class {label}' for label in unique_labels]
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(10, num_labels * 0.8), 
                                   max(8, num_labels * 0.7)))
    
    im = ax.imshow(averaged_rdm, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Dissimilarity', rotation=270, labelpad=20)
    
    # Add labels
    ax.set_xticks(range(num_labels))
    ax.set_xticklabels(label_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(num_labels))
    ax.set_yticklabels(label_names, fontsize=10)
    
    # Add values in cells if not too many labels
    if num_labels <= 20:
        for i in range(num_labels):
            for j in range(num_labels):
                text = ax.text(j, i, f'{averaged_rdm[i, j]:.2f}',
                             ha="center", va="center", color="w", fontsize=8)
    
    ax.set_title(f'Label-Averaged RDM: {layer_name}\n({FLAGS.distance_metric} distance)', 
                fontsize=12, pad=10)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f'  Saved label-averaged RDM to {output_path}')


def main(argv):
    del argv
    
    logging.info('='*80)
    logging.info('RDM Computation from MBT Activations')
    logging.info('='*80)
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Load activations
    logging.info('\n[1/5] Loading activations...')
    activations, sample_indices, available_layers = load_all_activations(FLAGS.activation_dir)
    num_samples = len(activations)
    
    # Load label mapping
    logging.info('\n[2/5] Loading label mapping...')
    label_to_name = load_label_mapping(FLAGS.activation_dir, FLAGS.label_mapping_file)
    
    # Get labels from logits
    logging.info('\n[3/5] Extracting labels from predictions...')
    labels = get_labels_from_logits(activations)
    unique_labels = np.unique(labels)
    logging.info(f'Found {len(unique_labels)} unique classes in {num_samples} samples')
    
    # Count samples per class
    for label in unique_labels:
        count = np.sum(labels == label)
        label_name = label_to_name.get(label, f'Class {label}')
        logging.info(f'  {label_name}: {count} samples')
    
    # Determine which layers to analyze
    if FLAGS.layers:
        layers_to_analyze = FLAGS.layers
        # Validate layers exist
        for layer in layers_to_analyze:
            if layer not in available_layers:
                raise ValueError(f'Layer {layer} not found in activations. '
                               f'Available: {available_layers}')
    else:
        # Analyze all encoder block outputs by default
        layers_to_analyze = [layer for layer in available_layers 
                            if 'encoder_block' in layer and 'output' in layer]
    
    logging.info(f'\nAnalyzing {len(layers_to_analyze)} layers:')
    for layer in layers_to_analyze:
        logging.info(f'  - {layer}')
    
    # Compute RDMs for each layer
    logging.info(f'\n[4/5] Computing RDMs ({FLAGS.distance_metric} distance)...')
    
    rdm_results = {}
    
    # Track what labels and counts we're using
    rdm_labels = labels  # Will be updated if averaging by class
    sample_counts = None
    
    for layer_name in layers_to_analyze:
        logging.info(f'\nProcessing layer: {layer_name}')
        
        # Extract activations for this layer
        layer_acts = extract_layer_activations(activations, layer_name)
        logging.info(f'  Activation shape: {layer_acts.shape}')
        
        # Average by class if requested
        if FLAGS.average_by_class:
            logging.info('  Averaging activations by class...')
            layer_acts, rdm_labels, sample_counts = average_activations_by_class(
                layer_acts, labels)
            logging.info(f'  Averaged activation shape: {layer_acts.shape}')
        
        # Compute RDM
        rdm = compute_rdm(layer_acts, 
                         metric=FLAGS.distance_metric,
                         standardize=FLAGS.standardize)
        logging.info(f'  RDM shape: {rdm.shape}')
        
        # Save RDM
        rdm_results[layer_name] = rdm
        
        # Save to file
        rdm_path = os.path.join(FLAGS.output_dir, f'rdm_{layer_name}.npz')
        save_data = {
            'rdm': rdm,
            'labels': rdm_labels,
            'layer_name': layer_name,
            'metric': FLAGS.distance_metric,
            'averaged_by_class': FLAGS.average_by_class
        }
        if sample_counts is not None:
            save_data['sample_counts'] = sample_counts
        
        np.savez_compressed(rdm_path, **save_data)
        logging.info(f'  Saved RDM to {rdm_path}')
    
    # Plot RDMs
    logging.info(f'\n[5/5] Plotting RDMs...')
    
    for layer_name, rdm in rdm_results.items():
        logging.info(f'\nPlotting {layer_name}...')
        
        # Plot full RDM (either sample-level or class-level depending on flag)
        plot_path = os.path.join(FLAGS.output_dir, f'rdm_{layer_name}.png')
        plot_rdm(rdm, rdm_labels, label_to_name, layer_name, plot_path, 
                FLAGS.plot_dendrograms)
        
        # Only compute label-averaged RDM if we haven't already averaged by class
        if not FLAGS.average_by_class:
            averaged_rdm, avg_labels = compute_label_averaged_rdm(rdm, rdm_labels)
            avg_plot_path = os.path.join(FLAGS.output_dir, 
                                        f'rdm_{layer_name}_averaged.png')
            plot_label_averaged_rdm(averaged_rdm, avg_labels, label_to_name, 
                                   layer_name, avg_plot_path)
        else:
            logging.info(f'  Skipping label-averaged plot (already averaged by class)')
    
    # Save summary
    summary = {
        'num_samples': num_samples,
        'num_classes': len(unique_labels),
        'layers_analyzed': layers_to_analyze,
        'distance_metric': FLAGS.distance_metric,
        'standardized': FLAGS.standardize,
        'averaged_by_class': FLAGS.average_by_class,
        'unique_labels': unique_labels.tolist(),
        'label_to_name': label_to_name
    }
    if sample_counts is not None:
        summary['sample_counts'] = sample_counts
    
    summary_path = os.path.join(FLAGS.output_dir, 'summary.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    logging.info('\n' + '='*80)
    logging.info('RDM Computation Complete!')
    logging.info(f'Processed {num_samples} samples across {len(unique_labels)} classes')
    logging.info(f'Analyzed {len(layers_to_analyze)} layers')
    logging.info(f'Outputs saved to: {FLAGS.output_dir}')
    logging.info('='*80)
    
    # Print usage instructions
    logging.info('\nTo load RDMs in Python:')
    logging.info('  import numpy as np')
    logging.info(f'  data = np.load("{FLAGS.output_dir}/rdm_encoder_block_L11_rgb_output.npz")')
    logging.info('  rdm = data["rdm"]')
    logging.info('  labels = data["labels"]')


if __name__ == '__main__':
    app.run(main)
