import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# --- Load ontology for semantic RDM plotting ---
with open("ontology.json", "r") as f:
    ontology = json.load(f)
id_to_node = {node["id"]: node for node in ontology}

# --- Plotting function from Semantic_RDM.py ---
def plot_semantic_rdm(rdm, class_ids, output_path=None, plot_dendrogram=False, minimal_plot=True, title=None):
    """
    Plot semantic RDM similar to compute_rdm_from_averaged_npy.py style.
    """
    num_classes = len(class_ids)
    
    # Get class names from ontology or use IDs
    class_labels = []
    for cid in class_ids:
        if isinstance(cid, np.integer):
            cid = int(cid)
        if isinstance(cid, int):
            # other class index, use labels_df
            name = labels_df[labels_df['index'] == cid]['display_name'].values
            if len(name) > 0:
                name = name[0]
            else:
                name = f"Class {cid}"
        else:
            # Semantic class ID (from ontology)
            if cid in id_to_node:
                name = id_to_node[cid]["name"]
            else:
                name = str(cid)
        if len(name) > 30:
            name = name[:27] + '...'
        class_labels.append(name)
    
    # Compute hierarchical clustering
    linkage_matrix = linkage(squareform(rdm), method='average')
    
    if plot_dendrogram:
        # Create figure with dendrograms on top and left
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
        
        ax_dendro_top = fig.add_subplot(gs[0, 1])
        ax_dendro_left = fig.add_subplot(gs[1, 0])
        ax_rdm = fig.add_subplot(gs[1, 1])
        
        # Top dendrogram
        dendro_top = dendrogram(linkage_matrix, ax=ax_dendro_top, no_labels=True)
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        for spine in ax_dendro_top.spines.values():
            spine.set_visible(False)
        
        # Left dendrogram
        dendro_left = dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left', no_labels=True)
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        for spine in ax_dendro_left.spines.values():
            spine.set_visible(False)
        
        # Reorder RDM by clustering
        order = dendro_top['leaves']
        rdm_sorted = rdm[order][:, order]
        labels_sorted = [class_labels[i] for i in order]
        
        # Plot RDM heatmap
        im = ax_rdm.imshow(rdm_sorted, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(im, ax=ax_rdm, fraction=0.046, pad=0.04)
        cbar.set_label('Dissimilarity', rotation=270, labelpad=20, fontsize=12)
        
        ax_rdm.set_xticks(range(num_classes))
        ax_rdm.set_xticklabels(labels_sorted, rotation=90, ha='center', fontsize=8, va='top')
        ax_rdm.set_yticks(range(num_classes))
        ax_rdm.set_yticklabels(labels_sorted, fontsize=8)
        
        if title is None:
            title = 'RDM'
        ax_rdm.set_title(title, fontsize=14, pad=15)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'Saved RDM plot to {output_path}')
        plt.close()
        
    elif minimal_plot:
        # Minimal plot: clustered heatmap without axis labels
        order = dendrogram(linkage_matrix, no_plot=True)['leaves']
        rdm_sorted = rdm[order][:, order]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(rdm_sorted, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dissimilarity')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        if title is None:
            title = 'Clustered RDM'
        ax.set_title(title, fontsize=14, pad=15)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f'Saved RDM plot to {output_path}')
        plt.close()
    
    return order if 'order' in locals() else None

# --- Load semantic RDM ---
semantic_data = np.load('Semantic_RDMs/semantic_rdm_all_classes.npz', allow_pickle=True)
semantic_rdm = semantic_data['rdm']
semantic_class_ids = semantic_data['class_ids']

print(f"Loaded semantic RDM with {len(semantic_class_ids)} classes")

# --- Load labels for mapping ---
labels_df = pd.read_csv('Video_csvs/audioset_labels.csv')
index_to_mid = dict(zip(labels_df['index'], labels_df['mid']))

# --- Find all other RDM files ---
#rdm_dir = 'RDM_from_averaged_CORRECT'
rdm_dir = 'RDM_from_averaged_CORRECT_Combined'
rdm_files = [f for f in os.listdir(rdm_dir) if f.startswith('rdm_') and f.endswith('.npz')]
print(f"Found {len(rdm_files)} other RDM files")

# --- Compare semantic RDM to all other RDMs ---
results = []
for rdm_file in rdm_files:
    other_data = np.load(os.path.join(rdm_dir, rdm_file), allow_pickle=True)
    other_rdm = other_data['rdm']
    other_class_indices = other_data['class_indices']
    
    # Map other indices to mids for alignment
    other_mids = [index_to_mid[idx] for idx in other_class_indices]
    
    # Find common classes
    common_mids = np.intersect1d(semantic_class_ids, other_mids)
    
    if len(common_mids) < 5:
        print(f"  {rdm_file}: Only {len(common_mids)} common classes, skipping")
        continue
    
    # Reindex both RDMs to common classes
    semantic_idx = [list(semantic_class_ids).index(mid) for mid in common_mids]
    other_idx = [other_mids.index(mid) for mid in common_mids]
    
    semantic_rdm_aligned = semantic_rdm[np.ix_(semantic_idx, semantic_idx)]
    other_rdm_aligned = other_rdm[np.ix_(other_idx, other_idx)]
    
    # Compute correlation
    rdm_vec_semantic = squareform(semantic_rdm_aligned)
    rdm_vec_other = squareform(other_rdm_aligned)
    corr, pval = spearmanr(rdm_vec_semantic, rdm_vec_other)
    
    results.append({
        'file': rdm_file,
        'correlation': corr,
        'pval': pval,
        'n_classes': len(common_mids),
        'semantic_rdm': semantic_rdm_aligned,
        'other_rdm': other_rdm_aligned,
        'class_indices': [int(i) for i in other_class_indices[[other_mids.index(mid) for mid in common_mids]]]
    })
    
    print(f"  {rdm_file}: r={corr:.4f}, p={pval:.4e}, n={len(common_mids)}")

# --- Find most correlated RDM ---
if not results:
    print("No valid comparisons found!")
    exit(1)

best_result = max(results, key=lambda x: x['correlation'])
print(f"\nMost correlated: {best_result['file']} with r={best_result['correlation']:.4f}")

# --- Plot semantic and best other RDM side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Plot semantic RDM
semantic_rdm_plot = best_result['semantic_rdm']
linkage_semantic = linkage(squareform(semantic_rdm_plot), method='average')
order_semantic = dendrogram(linkage_semantic, no_plot=True)['leaves']
rdm_semantic_sorted = semantic_rdm_plot[order_semantic][:, order_semantic]

im1 = ax1.imshow(rdm_semantic_sorted, cmap='viridis', aspect='auto')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Dissimilarity')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Semantic RDM (Clustered)', fontsize=16, pad=15)

# Plot other RDM in same order
other_rdm_plot = best_result['other_rdm']
rdm_other_sorted = other_rdm_plot[order_semantic][:, order_semantic]

im2 = ax2.imshow(rdm_other_sorted, cmap='viridis', aspect='auto')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Dissimilarity')
ax2.set_xticks([])
ax2.set_yticks([])
layer_name = best_result['file'].replace('rdm_', '').replace('.npz', '')
ax2.set_title(f'other RDM: {layer_name}\n(r={best_result["correlation"]:.4f}, p={best_result["pval"]:.4e})', 
              fontsize=16, pad=15)

plt.tight_layout()
comparison_path = os.path.join(rdm_dir, 'semantic_vs_best_other_comparison.png')
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison plot to {comparison_path}")
plt.close()

# --- Summary table ---
print("\n=== Summary of all comparisons ===")
for r in sorted(results, key=lambda x: x['correlation'], reverse=True):
    print(f"{r['file']:50s} | r={r['correlation']:7.4f} | p={r['pval']:.4e} | n={r['n_classes']}")