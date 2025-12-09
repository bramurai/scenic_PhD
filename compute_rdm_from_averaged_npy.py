#!/usr/bin/env python3
"""
Compute RDMs from per-class, per-layer averaged activations saved as .npy files.
Also generates RDM plots and dissimilarity evolution plots using the same plotting functions as compute_rdm_class_averaged.py.

Optoions: combined, recompute, plot_dendograms
Usage:
  python compute_rdm_from_averaged_npy.py \
    --averaged_dir=audioset_analysis_test/averaged_activations \
    --labels_csv=Video_csvs/audioset_labels.csv \
    --output_dir=RDM_test \
    --distance_metric=correlation \
    --minimal_plot \
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import argparse

# --- Plotting functions (adapted from compute_rdm_class_averaged.py) ---
def plot_rdm(rdm, class_indices, samples_per_class, index_to_name, layer_name, output_path, plot_dendrogram=True, minimal_plot=False):
    num_classes = len(class_indices)
    class_labels = []
    for idx in class_indices:
        name = index_to_name.get(idx, f'Class {idx}')
        count = samples_per_class.get(idx, 0)
        if len(name) > 30:
            name = name[:27] + '...'
        class_labels.append(f'{name} (n={count})')
    # Compute clustering and order
    linkage_matrix = linkage(squareform(rdm), method='average')
    if plot_dendrogram:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        ax_rdm = fig.add_subplot(gs[1, 1])
        dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True)
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        for spine in ax_dendro_top.spines.values():
            spine.set_visible(False)
        dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left', no_labels=True)
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        for spine in ax_dendro_left.spines.values():
            spine.set_visible(False)
        order = dendro_top['leaves']
        rdm_sorted = rdm[order][:, order]
        labels_sorted = [class_labels[i] for i in order]
        im = ax_rdm.imshow(rdm_sorted, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(im, ax=ax_rdm, fraction=0.046, pad=0.04)
        cbar.set_label('Dissimilarity', rotation=270, labelpad=20, fontsize=12)
        ax_rdm.set_xticks(range(num_classes))
        ax_rdm.set_xticklabels(labels_sorted, rotation=90, ha='center', fontsize=8, va='top')
        ax_rdm.set_yticks(range(num_classes))
        ax_rdm.set_yticklabels(labels_sorted, fontsize=8)
        ax_rdm.set_title(f'Class-Level RDM: {layer_name}\n({args.distance_metric} distance)', fontsize=14, pad=15)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved RDM plot to {output_path}')
    elif minimal_plot:
        # Minimal plot: reorder by clustering, no labels, no dendrograms
        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
        rdm_sorted = rdm[order][:, order]
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(rdm_sorted, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dissimilarity')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Clustered RDM: {layer_name}\n({args.distance_metric} distance)', fontsize=14, pad=15)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved minimal RDM plot to {output_path}')
    else:
        # Default: no clustering, no labels
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(rdm, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dissimilarity')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'RDM: {layer_name}\n({args.distance_metric} distance)', fontsize=14, pad=15)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved minimal RDM plot to {output_path}')

def plot_dissimilarity_evolution(rdm_results, class_indices, index_to_name, output_dir):
    import logging
    layer_names = list(rdm_results.keys())
    if len(layer_names) < 2:
        print('  Need at least 2 layers. Skipping evolution plot.')
        return
    num_classes = len(class_indices)
    mean_dissim, std_dissim, max_dissim, min_dissim = [], [], [], []
    within_class_dissim, between_class_dissim = [], []
    # For class-pair heatmap
    class_pairs = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            class_pairs.append((i, j))
    # Compute stats for each layer
    for layer_name in layer_names:
        rdm = rdm_results[layer_name]
        # Off-diagonal mask
        mask = ~np.eye(num_classes, dtype=bool)
        off_diag = rdm[mask]
        mean_dissim.append(np.mean(off_diag))
        std_dissim.append(np.std(off_diag))
        max_dissim.append(np.max(off_diag))
        min_dissim.append(np.min(off_diag))
        # Within-class: mean of diagonal (should be zero, but included for completeness)
        within = np.mean(np.diag(rdm))
        within_class_dissim.append(within)
        # Between-class: mean of off-diagonal
        between = np.mean(off_diag)
        between_class_dissim.append(between)
    # Clean up layer names for display, incrementing layer number by 1
    layer_labels = []
    for name in layer_names:
        if 'encoder_block_L' in name:
            parts = name.split('_')
            try:
                layer_num = int(parts[2][1:]) + 1  # L0 -> L1, L1 -> L2, etc.
            except Exception:
                layer_num = parts[2][1:]
            modality = parts[3] if len(parts) > 3 else ''
            layer_labels.append(f'L{layer_num} {modality}')
        else:
            layer_labels.append(name)
    x_pos = np.arange(len(layer_names))
    # Plot 1: Mean dissimilarity evolution and separation ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Top plot: Within vs Between class dissimilarity
    ax1.plot(x_pos, within_class_dissim, 'o-', label='Within-class', linewidth=2, markersize=8)
    ax1.plot(x_pos, between_class_dissim, 's-', label='Between-class', linewidth=2, markersize=8)
    ax1.plot(x_pos, mean_dissim, '^--', label='Overall mean', linewidth=2, markersize=8, alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax1.set_ylabel('Mean Dissimilarity', fontsize=11)
    ax1.set_title('Dissimilarity Evolution Across Layers', fontsize=13, pad=10)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Bottom plot: Separation ratio (between / within)
    separation_ratio = []
    for within, between in zip(within_class_dissim, between_class_dissim):
        if within > 1e-10:
            separation_ratio.append(between / within)
        else:
            separation_ratio.append(between / 1e-10 if between > 0 else 0)
    ax2.plot(x_pos, separation_ratio, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Separation threshold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(layer_labels, rotation=45, ha='right')
    ax2.set_ylabel('Separation Ratio\n(Between / Within)', fontsize=11)
    ax2.set_title('Class Separation Quality', fontsize=13, pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    evolution_path = os.path.join(output_dir, 'dissimilarity_evolution.png')
    #plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'  Saved evolution plot to {evolution_path}')

    # Plot 2: Heatmap of class-pair dissimilarity evolution (only if class-averaged and num_classes <= 30)
    if num_classes <= 30:
        fig, ax = plt.subplots(figsize=(max(12, len(layer_names) * 0.8), max(8, num_classes * 0.6)))
        # Create matrix: rows = class pairs, columns = layers
        class_pair_labels = []
        for i, j in class_pairs:
            name_i = index_to_name.get(class_indices[i], f'C{class_indices[i]}')
            name_j = index_to_name.get(class_indices[j], f'C{class_indices[j]}')
            # Truncate for readability
            if len(name_i) > 15:
                name_i = name_i[:12] + '...'
            if len(name_j) > 15:
                name_j = name_j[:12] + '...'
            class_pair_labels.append(f'{name_i} - {name_j}')
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
        ax.set_title('Class-Pair Dissimilarity Evolution Across Layers', fontsize=12, pad=10)
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'class_pair_evolution_heatmap.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f'  Saved class-pair heatmap to {heatmap_path}')

    # Plot 3: Mean absolute change in dissimilarity between consecutive layers (audio and rgb separately)
    # Identify audio and rgb layers
    def extract_layer_number(name):
        # Extracts the integer after '_L' in the layer name
        if '_L' in name:
            try:
                return int(name.split('_L')[1].split('_')[0])
            except Exception:
                return -1
        return -1

    # Get and sort audio/rgb layer indices by layer number
    audio_idxs = sorted([i for i, name in enumerate(layer_names) if 'audio' in name.lower()], key=lambda i: extract_layer_number(layer_names[i]))
    rgb_idxs = sorted([i for i, name in enumerate(layer_names) if 'rgb' in name.lower()], key=lambda i: extract_layer_number(layer_names[i]))

    def plot_mean_abs_change(idx_list, label, fname):
        if len(idx_list) < 2:
            return
        abs_changes = []
        transition_labels = []
        for i in range(1, len(idx_list)):
            prev_idx = idx_list[i-1]
            curr_idx = idx_list[i]
            rdm_prev = rdm_results[layer_names[prev_idx]]
            rdm_curr = rdm_results[layer_names[curr_idx]]
            mask = ~np.eye(num_classes, dtype=bool)
            diff = np.abs(rdm_curr - rdm_prev)
            abs_changes.append(np.mean(diff[mask]))
            transition_labels.append(f'{layer_labels[prev_idx]}→{layer_labels[curr_idx]}')
        fig, ax = plt.subplots(figsize=(max(12, len(abs_changes) * 0.7), 6))
        ax.plot(range(1, len(idx_list)), abs_changes, marker='o', linewidth=2, label=f'Mean |Δ dissimilarity| ({label})')
        ax.set_xticks(range(1, len(idx_list)))
        ax.set_xticklabels(transition_labels, rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Layer Transition', fontsize=12)
        ax.set_ylabel('Mean |Δ Dissimilarity|', fontsize=12)
        ax.set_title(f'Mean Absolute Change in Dissimilarity ({label} layers)', fontsize=14, pad=15)
        # Add red vertical line after L8 if present
        fusion_idx = None
        for i in range(1, len(idx_list)):
            l_prev = layer_labels[idx_list[i-1]]
            l_next = layer_labels[idx_list[i]]
            # Find transition between L8 and L9 (now labeled as L9 and L10)
            if (l_prev.startswith('L8')) and (l_next.startswith('L9')):
                fusion_idx = i - 0.5  # Place line between L9 and L10
                break
        if fusion_idx is not None:
            ax.axvline(fusion_idx, color='red', linestyle='--', linewidth=2)
            ax.text(fusion_idx+0.1, ax.get_ylim()[1]*0.95, 'fusion at L8', color='red', rotation=90, va='top', ha='left', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        change_path = os.path.join(output_dir, fname)
        plt.savefig(change_path, dpi=150, bbox_inches='tight')
        plt.close()
        import logging
        logging.info(f'  Saved mean absolute change plot to {change_path}')

    plot_mean_abs_change(audio_idxs, 'audio', 'mean_abs_change_dissimilarity_audio.png')
    plot_mean_abs_change(rgb_idxs, 'rgb', 'mean_abs_change_dissimilarity_rgb.png')

# --- Main script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--averaged_dir', required=True, help='Directory with per-class, per-layer averaged .npy files')
    parser.add_argument('--labels_csv', required=True, help='CSV file with class index and display_name columns')
    parser.add_argument('--output_dir', default='RDM_from_averaged', help='Output directory')
    parser.add_argument('--distance_metric', default='correlation', choices=['correlation', 'euclidean', 'cosine', 'cityblock'])
    parser.add_argument('--standardize', action='store_true', help='Standardize activations before computing distances')
    parser.add_argument('--plot_dendrograms', action='store_true', help='Plot dendrograms in RDM plots')
    parser.add_argument('--minimal_plot', action='store_true', help='Plot only clustered RDM heatmap without axis labels or dendrograms')
    parser.add_argument('--recompute', action='store_true', help='Recompute RDMs even if output files already exist')
    parser.add_argument('--combined', action='store_true', help='For each layer, concatenate RGB and audio activations for each class and compute a single RDM per layer')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels_df = pd.read_csv(args.labels_csv)
    index_to_name = dict(zip(labels_df['index'], labels_df['display_name']))
    files = [f for f in os.listdir(args.averaged_dir) if f.endswith('.npy')]
    layer_to_classfiles = {}
    for f in files:
        parts = f.split('_', 2)
        if len(parts) < 3:
            continue
        class_idx = int(parts[1])
        layer_name = parts[2].replace('.npy', '')
        layer_to_classfiles.setdefault(layer_name, []).append((class_idx, f))

    # Group layers by base name (e.g., encoder_block_L0_audio, encoder_block_L0_rgb)
    from collections import defaultdict
    base_to_layer = defaultdict(dict)
    for layer_name in layer_to_classfiles:
        print(layer_name[:-7])
        if layer_name[:-7].endswith('_audio'):
            base = layer_name[:-13]
            base_to_layer[base]['audio'] = layer_name
        elif layer_name[:-7].endswith('_rgb'):
            base = layer_name[:-11]
            base_to_layer[base]['rgb'] = layer_name

    rdm_results = {}
    print(base_to_layer.items())
    for base, mods in base_to_layer.items():
        # Only process if both modalities exist for this base layer
        if args.combined and ('audio' in mods and 'rgb' in mods):
            audio_classfiles = sorted(layer_to_classfiles[mods['audio']])
            rgb_classfiles = sorted(layer_to_classfiles[mods['rgb']])
            # Find common class indices
            audio_classes = set(idx for idx, _ in audio_classfiles)
            rgb_classes = set(idx for idx, _ in rgb_classfiles)
            common_classes = sorted(audio_classes & rgb_classes)
            if not common_classes:
                continue
            # Build concatenated activations for each class
            activations = []
            for idx in common_classes:
                audio_file = [f for i, f in audio_classfiles if i == idx][0]
                rgb_file = [f for i, f in rgb_classfiles if i == idx][0]
                audio_act = np.load(os.path.join(args.averaged_dir, audio_file)).flatten()
                rgb_act = np.load(os.path.join(args.averaged_dir, rgb_file)).flatten()
                activations.append(np.concatenate([audio_act, rgb_act]))
            activations = np.stack(activations)
            if args.standardize:
                activations = StandardScaler().fit_transform(activations)
            rdm = squareform(pdist(activations, metric=args.distance_metric))
            rdm_npz_path = os.path.join(args.output_dir, f'rdm_{base}_combined.npz')
            rdm_png_path = os.path.join(args.output_dir, f'rdm_{base}_combined.png')
            np.savez_compressed(
                rdm_npz_path,
                rdm=rdm,
                class_indices=np.array(common_classes),
                class_names=np.array([index_to_name.get(idx, f'Class {idx}') for idx in common_classes])
            )
            print(f'Saved combined RDM for {base} to {rdm_npz_path}')
            samples_per_class = {idx: 0 for idx in common_classes}
            plot_rdm(
                rdm, common_classes, samples_per_class, index_to_name, f'{base}_combined',
                rdm_png_path,
                plot_dendrogram=args.plot_dendrograms, minimal_plot=args.minimal_plot)
            rdm_results[f'{base}_combined'] = rdm
        elif not args.combined:
            # Per-modality RDMs as before
            for modality in mods:
                layer_name = mods[modality]
                classfiles = sorted(layer_to_classfiles[layer_name])
                class_indices = [idx for idx, _ in classfiles]
                rdm_npz_path = os.path.join(args.output_dir, f'rdm_{layer_name}.npz')
                rdm_png_path = os.path.join(args.output_dir, f'rdm_{layer_name}.png')
                if (not args.recompute) and os.path.exists(rdm_npz_path):
                    print(f'Skipping {layer_name}: {rdm_npz_path} already exists. Use --recompute to overwrite.')
                    if not os.path.exists(rdm_png_path):
                        data = np.load(rdm_npz_path, allow_pickle=True)
                        rdm = data['rdm']
                        samples_per_class = {idx: 0 for idx in class_indices}
                        plot_rdm(
                            rdm, class_indices, samples_per_class, index_to_name, layer_name,
                            rdm_png_path,
                            plot_dendrogram=args.plot_dendrograms, minimal_plot=args.minimal_plot)
                    data = np.load(rdm_npz_path, allow_pickle=True)
                    rdm_results[layer_name] = data['rdm']
                    continue
                activations = []
                for idx, fname in classfiles:
                    act = np.load(os.path.join(args.averaged_dir, fname)).flatten()
                    activations.append(act)
                activations = np.stack(activations)
                if args.standardize:
                    activations = StandardScaler().fit_transform(activations)
                rdm = squareform(pdist(activations, metric=args.distance_metric))
                rdm_results[layer_name] = rdm
                np.savez_compressed(
                    rdm_npz_path,
                    rdm=rdm,
                    class_indices=np.array(class_indices),
                    class_names=np.array([index_to_name.get(idx, f'Class {idx}') for idx in class_indices])
                )
                print(f'Saved RDM for {layer_name} to {rdm_npz_path}')
                samples_per_class = {idx: 0 for idx in class_indices}
                plot_rdm(
                    rdm, class_indices, samples_per_class, index_to_name, layer_name,
                    rdm_png_path,
                    plot_dendrogram=args.plot_dendrograms, minimal_plot=args.minimal_plot)
    # Always update advanced evolution plots, even if RDMs were skipped
    if rdm_results:
        # Use the last computed class_indices for evolution plot
        plot_dissimilarity_evolution(rdm_results, class_indices, index_to_name, args.output_dir)
