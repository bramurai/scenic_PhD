#!/usr/bin/env python3
"""Analyze how RDMs change across layers and visualize the differences."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load RDMs from all layers
layers = [f'L{i}' for i in range(12)]
modalities = ['audio', 'rgb']

print('='*80)
print('Detailed RDM Change Analysis')
print('='*80)

# 1. Create correlation matrix showing similarity between all layer pairs
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for mod_idx, modality in enumerate(modalities):
    rdms = {}
    for layer in layers:
        rdm_file = f'ARDM_class_averaged/rdm_encoder_block_{layer}_{modality}_output.npz'
        rdms[layer] = np.load(rdm_file)['rdm']
    
    # Compute correlation matrix between all layer pairs
    n_layers = len(layers)
    corr_matrix = np.zeros((n_layers, n_layers))
    
    for i, layer_i in enumerate(layers):
        for j, layer_j in enumerate(layers):
            rdm_i = rdms[layer_i].flatten()
            rdm_j = rdms[layer_j].flatten()
            corr_matrix[i, j] = np.corrcoef(rdm_i, rdm_j)[0, 1]
    
    # Plot
    im = axes[mod_idx].imshow(corr_matrix, cmap='RdYlGn', vmin=0.7, vmax=1.0)
    axes[mod_idx].set_xticks(range(n_layers))
    axes[mod_idx].set_xticklabels(layers, rotation=45, ha='right')
    axes[mod_idx].set_yticks(range(n_layers))
    axes[mod_idx].set_yticklabels(layers)
    axes[mod_idx].set_title(f'{modality.upper()} RDM Correlations Across Layers', 
                           fontsize=13, fontweight='bold')
    axes[mod_idx].set_xlabel('Layer')
    axes[mod_idx].set_ylabel('Layer')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[mod_idx], fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values
    for i in range(n_layers):
        for j in range(n_layers):
            text = axes[mod_idx].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                     ha="center", va="center", 
                                     color="black" if corr_matrix[i, j] > 0.85 else "white",
                                     fontsize=7)

plt.tight_layout()
plt.savefig('ARDM_class_averaged/rdm_layer_correlations.png', dpi=150, bbox_inches='tight')
print('\n✓ Saved: ARDM_class_averaged/rdm_layer_correlations.png')
plt.close()

# 2. Show actual RDM differences between selected layers
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

comparisons = [
    ('L0', 'L6', 'Early → Mid'),
    ('L6', 'L11', 'Mid → Late'),
    ('L0', 'L11', 'Early → Late')
]

for mod_idx, modality in enumerate(modalities):
    rdms = {}
    for layer in layers:
        rdm_file = f'ARDM_class_averaged/rdm_encoder_block_{layer}_{modality}_output.npz'
        rdms[layer] = np.load(rdm_file)['rdm']
    
    for comp_idx, (layer_a, layer_b, title) in enumerate(comparisons):
        ax = axes[mod_idx, comp_idx]
        
        # Compute difference
        diff = rdms[layer_b] - rdms[layer_a]
        
        # Plot
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'{modality.upper()}: {title}\n({layer_a} → {layer_b})', 
                    fontsize=11, fontweight='bold')
        
        if comp_idx == 0:
            ax.set_ylabel('Class', fontsize=10)
        if mod_idx == 1:
            ax.set_xlabel('Class', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if comp_idx == 2:
            cbar.set_label('Δ Dissimilarity', rotation=270, labelpad=20)
        
        # Add statistics
        mean_change = np.abs(diff).mean()
        max_change = np.abs(diff).max()
        ax.text(0.02, 0.98, f'Mean |Δ|: {mean_change:.3f}\nMax |Δ|: {max_change:.3f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('ARDM_class_averaged/rdm_layer_differences.png', dpi=150, bbox_inches='tight')
print('✓ Saved: ARDM_class_averaged/rdm_layer_differences.png')
plt.close()

# 3. Quantify representation change per layer
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for mod_idx, modality in enumerate(modalities):
    rdms = {}
    for layer in layers:
        rdm_file = f'ARDM_class_averaged/rdm_encoder_block_{layer}_{modality}_output.npz'
        rdms[layer] = np.load(rdm_file)['rdm']
    
    # Compute change from previous layer
    changes = []
    for i in range(1, len(layers)):
        diff = np.abs(rdms[layers[i]] - rdms[layers[i-1]]).mean()
        changes.append(diff)
    
    ax = axes[mod_idx]
    x = range(1, len(layers))
    ax.plot(x, changes, 'o-', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{layers[i-1]}→{layers[i]}' for i in x], rotation=45, ha='right')
    ax.set_xlabel('Layer Transition', fontsize=11)
    ax.set_ylabel('Mean Absolute Change', fontsize=11)
    ax.set_title(f'{modality.upper()}: Representation Change per Layer', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight fusion layer (L8)
    if len(changes) > 7:
        ax.axvline(x=7.5, color='red', linestyle='--', alpha=0.5, label='Fusion at L8')
        ax.legend()

plt.tight_layout()
plt.savefig('ARDM_class_averaged/rdm_change_per_layer.png', dpi=150, bbox_inches='tight')
print('✓ Saved: ARDM_class_averaged/rdm_change_per_layer.png')
plt.close()

print('\n' + '='*80)
print('Analysis complete! Created 3 visualizations:')
print('1. rdm_layer_correlations.png - Shows similarity between all layer pairs')
print('2. rdm_layer_differences.png - Shows actual differences between key layers')
print('3. rdm_change_per_layer.png - Shows how much representation changes per layer')
print('='*80)
