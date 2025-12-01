import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import pandas as pd

# ---------------------------------------------------------
# 1. Load AudioSet Ontology (ontology.json)
#    Download from: https://github.com/audioset/ontology
# ---------------------------------------------------------

with open("ontology.json", "r") as f:
    ontology = json.load(f)
print(f"Loaded ontology with {len(ontology)} nodes.")
# Build a dict from ID -> node
id_to_node = {node["id"]: node for node in ontology}

# ---------------------------------------------------------
# 2. Build directed graph of the ontology
# ---------------------------------------------------------

G = nx.DiGraph()

for node in ontology:
    nid = node["id"]
    G.add_node(nid)
    for child in node.get("child_ids", []):
        G.add_edge(nid, child)  # parent -> child

# Also add reverse edges for shortest-path in undirected form
UG = G.to_undirected()

# ---------------------------------------------------------
# 3. Helper: compute lowest-common-ancestor depth
# ---------------------------------------------------------

# Precompute depths from the root ("/m/09x0r") — the ontology root
# But AudioSet has multiple top-level roots, so we compute min depth among any root.
def compute_depths(graph):
    depths = {}
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]

    for root in roots:
        for node, d in nx.single_source_shortest_path_length(graph, root).items():
            if node not in depths:
                depths[node] = d
            else:
                depths[node] = min(depths[node], d)
    return depths

depth = compute_depths(G)

def lca_depth(a, b):
    # get all ancestors (including self)
    ancestors_a = set(nx.ancestors(G, a)) | {a}
    ancestors_b = set(nx.ancestors(G, b)) | {b}

    common = ancestors_a & ancestors_b
    if not common:
        return 0
    return max(depth[x] for x in common)

# ---------------------------------------------------------
# 4. Build a semantic RDM for selected classes
# ---------------------------------------------------------

def build_semantic_rdm(class_ids, metric="shortest_path"):
    n = len(class_ids)
    rdm = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            a, b = class_ids[i], class_ids[j]

            if metric == "shortest_path":
                d = nx.shortest_path_length(UG, a, b)
            
            elif metric == "lca":
                # Semantic distance = (depth(a) + depth(b)) - 2 * depth(LCA)
                # Clamp to non-negative to avoid hierarchical clustering errors
                d = max(0, depth[a] + depth[b] - 2 * lca_depth(a, b))

            else:
                raise ValueError("Unknown metric")

            rdm[i, j] = d
            rdm[j, i] = d
    return rdm

def normalize_rdm(rdm):
    """Normalize RDM to [0, 1] range using min-max normalization."""
    rdm_min = rdm.min()
    rdm_max = rdm.max()
    if rdm_max - rdm_min > 0:
        rdm_normalized = (rdm - rdm_min) / (rdm_max - rdm_min)
    else:
        rdm_normalized = rdm  # All values are the same
    return rdm_normalized

def plot_semantic_rdm(rdm, class_ids, output_path=None, plot_dendrogram=False, minimal_plot=True, title=None):
    """
    Plot semantic RDM similar to compute_rdm_from_averaged_npy.py style.
    
    Args:
        rdm: RDM matrix (n x n)
        class_ids: List of class IDs (strings like "/m/04rlf")
        output_path: Path to save figure (if None, display only)
        plot_dendrogram: If True, include dendrograms on top and left
        minimal_plot: If True, plot only clustered heatmap without labels
        title: Custom title for plot
    """
    num_classes = len(class_ids)
    
    # Get class names from ontology
    class_labels = []
    for cid in class_ids:
        if cid in id_to_node:
            name = id_to_node[cid]["name"]
        else:
            name = cid
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
            title = 'Semantic RDM (AudioSet Ontology)'
        ax_rdm.set_title(title, fontsize=14, pad=15)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
            title = 'Clustered Semantic RDM (AudioSet Ontology)'
        ax.set_title(title, fontsize=14, pad=15)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    else:
        # Default: RDM without clustering or dendrograms
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(rdm, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Dissimilarity')
        
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_labels, rotation=90, ha='center', fontsize=8, va='top')
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(class_labels, fontsize=8)
        
        if title is None:
            title = 'Semantic RDM (AudioSet Ontology)'
        ax.set_title(title, fontsize=14, pad=15)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    if output_path:
        print(f'Saved semantic RDM plot to {output_path}')




# ---------------------------------------------------------
# 6. Plot and save RDM for all AudioSet classes
# ---------------------------------------------------------

labels_df = pd.read_csv("Video_csvs/audioset_labels.csv")
all_class_ids = labels_df["mid"].tolist()

valid_class_ids = [cid for cid in all_class_ids if cid in depth]
print(f"Using {len(valid_class_ids)} valid class IDs out of {len(all_class_ids)}")

rdm = build_semantic_rdm(valid_class_ids, metric="lca")
print("Any negative values in RDM before normalization?", np.any(rdm < 0))

# Normalize RDM to [0, 1]
rdm_normalized = normalize_rdm(rdm)
print("RDM min:", rdm_normalized.min(), "max:", rdm_normalized.max())
print("Any negative values in normalized RDM?", np.any(rdm_normalized < 0))
# After computing rdm_normalized, add:
np.savez_compressed(
    'Semantic_RDMs/semantic_rdm_all_classes.npz',
    rdm=rdm_normalized,
    class_ids=np.array(valid_class_ids),
    class_names=np.array([id_to_node[cid]["name"] if cid in id_to_node else cid for cid in valid_class_ids])
)
print("Saved semantic RDM with metadata to Semantic_RDMs/semantic_rdm_all_classes.npz")
# Save RDM
np.save('Semantic_RDMs/semantic_rdm_all_classes.npy', rdm_normalized)

print("Saved normalized RDM to Semantic_RDMs/semantic_rdm_all_classes.npy")

# Plot minimal (clustered but no labels)
plot_semantic_rdm(rdm_normalized, valid_class_ids, output_path='Semantic_RDMs/semantic_rdm_minimal.png', 
                  minimal_plot=True, title='Semantic RDM (Clustered, Normalized)')
