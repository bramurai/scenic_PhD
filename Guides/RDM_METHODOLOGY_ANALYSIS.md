# RDM Computation and Comparison Methodology - Analysis

## Executive Summary

**Overall Assessment: The methodology is sound and well-designed.** ✅

The pipeline correctly:
1. Computes neural RDMs from class-averaged MLP activations using correlation distance
2. Computes semantic RDMs from AudioSet ontology using tree-based distances
3. Compares them via Spearman correlation on aligned class subsets
4. Applies proper visualization with hierarchical clustering

---

## Part 1: Neural RDM Computation (`compute_rdm_from_averaged_npy.py`)

### Data Input
```python
# For each layer and class:
act = np.load(os.path.join(args.averaged_dir, fname)).flatten()
activations.append(act)
activations = np.stack(activations)  # Shape: (n_classes, n_features)
```

**What happens:**
- Loads class-averaged activations (already averaged across all samples per class)
- **Flattens** each activation to a 1D vector (collapses spatial/temporal dimensions)
- Stacks into matrix: rows = classes, columns = features

**Is this correct?** ✅ **YES**
- Flattening is standard for RDM computation - treats each position in the feature map as an independent dimension
- For MLP output with shape `[seq_len, hidden_dim]`, flattening gives `[seq_len × hidden_dim]` feature vector
- This captures the full representational geometry across all positions

### Optional Standardization
```python
if args.standardize:
    activations = StandardScaler().fit_transform(activations)
```

**What it does:**
- Centers each feature dimension to mean=0, std=1 across classes
- Removes scale differences between feature dimensions

**Is this correct?** ✅ **YES, but optional**
- **Without standardization**: Features with larger magnitudes dominate distance computation
- **With standardization**: All features contribute equally
- For correlation distance (default), standardization has **minimal effect** since correlation already normalizes
- For Euclidean distance, standardization is **highly recommended**

### Distance Computation
```python
rdm = squareform(pdist(activations, metric=args.distance_metric))
```

**Default metric: `'correlation'`**

For two class activation vectors $\mathbf{a}$ and $\mathbf{b}$:

$$\text{correlation\_distance}(\mathbf{a}, \mathbf{b}) = 1 - \text{Pearson}(\mathbf{a}, \mathbf{b})$$

Where Pearson correlation is:

$$\text{Pearson}(\mathbf{a}, \mathbf{b}) = \frac{\text{cov}(\mathbf{a}, \mathbf{b})}{\sigma_\mathbf{a} \sigma_\mathbf{b}}$$

**Why correlation distance?** ✅ **Excellent choice!**
- **Invariant to scale**: Captures similarity in *patterns*, not magnitudes
- **Standard in neuroscience**: RSA (Representational Similarity Analysis) literature uses correlation
- **Interpretable**: Distance=0 means perfect correlation, Distance=2 means perfect anti-correlation
- **Insensitive to activation magnitude**: Two classes with similar response patterns but different overall firing rates are still considered similar

**Alternative metrics available:**
- `euclidean`: $\|\mathbf{a} - \mathbf{b}\|_2$ - sensitive to magnitude differences
- `cosine`: $1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ - similar to correlation but doesn't center
- `cityblock`: $\sum_i |a_i - b_i|$ - L1 distance

### Combined Modality Mode
```python
if args.combined and ('audio' in mods and 'rgb' in mods):
    audio_act = np.load(...).flatten()
    rgb_act = np.load(...).flatten()
    activations.append(np.concatenate([audio_act, rgb_act]))
```

**What it does:**
- Concatenates audio and RGB activations for the same layer
- Creates joint representation: `[audio_features, rgb_features]`

**Is this correct?** ✅ **YES - clever approach!**
- Captures **multimodal representations** at each layer
- Allows measuring how audio+visual information is jointly organized
- Especially interesting **after fusion layers** (L8+) where modalities interact

---

## Part 2: Semantic RDM Computation (`Semantic_RDM.py`)

### Ontology Structure
```python
G = nx.DiGraph()  # Directed graph
for node in ontology:
    for child in node.get("child_ids", []):
        G.add_edge(nid, child)  # parent -> child

UG = G.to_undirected()  # For shortest path computation
```

**Structure:**
- AudioSet ontology is a **directed acyclic graph (DAG)** with multiple roots
- Edges represent **is-a** relationships (e.g., "Dog bark" → "Animal")
- Converted to undirected for distance computation

### Distance Metric: LCA (Lowest Common Ancestor)
```python
def lca_depth(a, b):
    ancestors_a = set(nx.ancestors(G, a)) | {a}
    ancestors_b = set(nx.ancestors(G, b)) | {b}
    common = ancestors_a & ancestors_b
    return max(depth[x] for x in common)

# Semantic distance formula:
d = max(0, depth[a] + depth[b] - 2 * lca_depth(a, b))
```

**Mathematical interpretation:**

For two concepts $a$ and $b$ in the ontology tree:

$$\text{semantic\_distance}(a, b) = \text{depth}(a) + \text{depth}(b) - 2 \times \text{depth}(\text{LCA}(a, b))$$

**Intuition:**
- **Close relatives** (share recent common ancestor): Low distance
  - Example: "Dog bark" and "Cat meow" → LCA = "Animal sounds" → Small distance
- **Distant relatives** (only share root): High distance
  - Example: "Dog bark" and "Violin" → LCA = "Root" → Large distance

**Visualization:**
```
        Root (depth=0)
        /    \
   Animal   Music (depth=1)
      |       |
     Dog   Violin (depth=2)
     
distance(Dog, Violin) = 2 + 2 - 2*0 = 4
distance(Dog, Animal) = 2 + 1 - 2*1 = 1
```

**Is this correct?** ✅ **YES - standard tree-based semantic distance**
- Well-established in computational linguistics and ontology research
- Corresponds to "path length in tree" accounting for hierarchy
- The `max(0, ...)` clamp prevents negative values from graph inconsistencies

### Normalization
```python
rdm_normalized = (rdm - rdm_min) / (rdm_max - rdm_min)
```

**What it does:** Scales semantic distances to [0, 1] range

**Is this correct?** ✅ **YES - necessary for comparison**
- Neural RDMs use correlation distance (range [0, 2] but typically [0, 1.5])
- Semantic RDMs use integer path lengths (range depends on tree depth)
- **Normalization makes them comparable** via Spearman correlation
- Min-max normalization preserves rank order (important for Spearman)

---

## Part 3: RDM Comparison (`RDM_compare.py`)

### Class Alignment Strategy
```python
# Neural RDM uses integer indices (0-526)
other_class_indices = other_data['class_indices']

# Semantic RDM uses AudioSet MIDs ("/m/04rlf")
semantic_class_ids = semantic_data['class_ids']

# Map neural indices to MIDs for alignment
other_mids = [index_to_mid[idx] for idx in other_class_indices]

# Find common classes
common_mids = np.intersect1d(semantic_class_ids, other_mids)

# Reindex both RDMs to common classes only
semantic_idx = [list(semantic_class_ids).index(mid) for mid in common_mids]
other_idx = [other_mids.index(mid) for mid in common_mids]

semantic_rdm_aligned = semantic_rdm[np.ix_(semantic_idx, semantic_idx)]
other_rdm_aligned = other_rdm[np.ix_(other_idx, other_idx)]
```

**What happens:**
1. Neural RDM classes identified by integer indices (e.g., 137)
2. Semantic RDM classes identified by MIDs (e.g., "/m/04rlf")
3. **Mapping via `labels_df`**: Converts indices → MIDs
4. **Intersection**: Finds classes present in BOTH RDMs
5. **Reindexing**: Extracts aligned submatrices in matching order

**Is this correct?** ✅ **ABSOLUTELY CRITICAL and well-done!**
- RDMs must have **same classes in same order** for valid comparison
- Using `np.ix_(semantic_idx, semantic_idx)` correctly extracts submatrix
- Handles case where neural RDM might be missing some classes (data issues)

### Comparison Metric: Spearman Correlation
```python
rdm_vec_semantic = squareform(semantic_rdm_aligned)
rdm_vec_other = squareform(other_rdm_aligned)
corr, pval = spearmanr(rdm_vec_semantic, rdm_vec_other)
```

**What it does:**
1. **`squareform()`**: Extracts upper triangle of RDM as 1D vector
   - RDM shape: (n, n) → Vector shape: (n*(n-1)/2,)
   - Removes redundant lower triangle and diagonal
2. **Spearman correlation**: Rank-based correlation between the two RDM vectors

**Why Spearman instead of Pearson?** ✅ **Correct choice!**
- **Spearman**: Measures monotonic relationship (rank correlation)
  - Insensitive to nonlinear transformations
  - Robust to outliers
  - **Standard in RSA literature**
- **Pearson**: Measures linear relationship only
  - Would be affected by the different scales/metrics used
  - Less robust

**Mathematical interpretation:**

$$\rho_{\text{Spearman}} = \text{Pearson}(\text{rank}(\mathbf{x}), \text{rank}(\mathbf{y}))$$

Where $\rho = 1$ means perfect rank agreement, $\rho = 0$ means no relationship.

**Statistical significance:** The p-value tests $H_0: \rho = 0$ (no correlation)

### Visualization Strategy
```python
# 1. Cluster semantic RDM
linkage_semantic = linkage(squareform(semantic_rdm_plot), method='average')
order_semantic = dendrogram(linkage_semantic, no_plot=True)['leaves']

# 2. Apply same ordering to both RDMs
rdm_semantic_sorted = semantic_rdm_plot[order_semantic][:, order_semantic]
rdm_other_sorted = other_rdm_plot[order_semantic][:, order_semantic]

# 3. Plot side-by-side
im1 = ax1.imshow(rdm_semantic_sorted, ...)
im2 = ax2.imshow(rdm_other_sorted, ...)
```

**What it does:**
- Clusters semantic RDM to find natural groupings
- **Applies semantic clustering order to neural RDM**
- Enables visual comparison of structure alignment

**Is this correct?** ✅ **BRILLIANT!**
- Reveals whether neural RDM respects semantic category boundaries
- If highly correlated, neural RDM should show similar block structure
- Makes visual pattern matching easy

---

## Methodological Strengths

### 1. ✅ Appropriate Distance Metrics
- **Neural**: Correlation distance (pattern-based, scale-invariant)
- **Semantic**: Tree path length (ontologically grounded)
- Both are well-justified for their respective domains

### 2. ✅ Proper Normalization
- Semantic RDM normalized to [0, 1] for comparability
- Rank-based comparison (Spearman) makes absolute scales less critical

### 3. ✅ Robust Statistical Testing
- Spearman correlation with p-values
- Handles non-normal distributions and nonlinear relationships

### 4. ✅ Careful Class Alignment
- Explicit mapping via MIDs
- Submatrix extraction preserves pairwise structure
- Validates sufficient overlap (requires ≥5 common classes)

### 5. ✅ Multimodal Analysis
- `--combined` mode captures cross-modal representations
- Allows tracking fusion at L8 layer

### 6. ✅ Visualization Quality
- Hierarchical clustering reveals structure
- Side-by-side comparison with shared ordering
- Minimal plots avoid clutter while preserving information

---

## Potential Considerations

### 1. ⚠️ Flattening Loses Spatial Information
**Current approach:**
```python
act.flatten()  # Collapses [seq_len, hidden_dim] → [seq_len * hidden_dim]
```

**Implication:**
- Treats position 0 and position 197 as independent dimensions
- Loses explicit spatial/temporal structure

**Is this a problem?** 🤔 **Depends on research question**
- **For global representational geometry**: Flattening is standard and appropriate
- **For position-specific analysis**: Could preserve structure and compute RDM per position

**Alternative approach (not implemented):**
```python
# Average over positions first
act_mean = act.mean(axis=0)  # Shape: [hidden_dim]
```
This would capture "what features are active" rather than "where they're active."

### 2. ⚠️ Semantic Distance Choices
**Current metric:** LCA tree-based distance

**Alternative metrics:**
- **Information Content**: Weight nodes by frequency in dataset
  $$\text{IC}(c) = -\log P(\text{class } c)$$
  
- **Wu-Palmer Similarity**: 
  $$\text{sim}(a,b) = \frac{2 \times \text{depth}(\text{LCA}(a,b))}{\text{depth}(a) + \text{depth}(b)}$$

**Current choice is reasonable**, but these alternatives might capture semantic similarity better for unbalanced trees.

### 3. ⚠️ Multiple Comparison Correction
**Current:** Tests correlation for each layer independently

**Issue:** Testing 24 layers (L0-L11 × audio/RGB) inflates false positive rate

**Solution (not critical but good practice):**
```python
from statsmodels.stats.multitest import multipletests
corrected_pvals = multipletests(pvals, method='bonferroni')[1]
```

### 4. ⚠️ Sample Size Per Class
**Not currently tracked** in comparison, but important because:
- Classes with 1-2 samples: High variance in averaged activations
- Classes with 100+ samples: Stable, reliable averages

**Potential improvement:**
```python
# Weight correlation by reliability
weights = np.sqrt([n_samples_i * n_samples_j for i, j in class_pairs])
weighted_corr = weighted_spearmanr(rdm_vec_semantic, rdm_vec_other, weights)
```

---

## Answers to Key Questions

### Q1: Does neural RDM computation make sense?
**YES** ✅
- Correlation distance on flattened class-averaged activations is standard RSA methodology
- Captures representational geometry independent of scale
- Combined modality mode intelligently handles multimodal fusion

### Q2: Does semantic RDM computation make sense?
**YES** ✅
- LCA-based tree distance is established semantic similarity metric
- AudioSet ontology provides ground truth hierarchical structure
- Normalization to [0,1] makes comparison possible

### Q3: Does the comparison methodology make sense?
**YES** ✅
- Spearman correlation is the right choice (robust, rank-based)
- Class alignment via MIDs is correctly implemented
- Visualization strategy (shared clustering order) is insightful

### Q4: Are there any serious flaws?
**NO** ❌
- The methodology is sound and follows best practices
- Minor enhancements possible (see considerations) but not critical

---

## Interpretation Guide

### What does high correlation mean?
- **r = 0.7-1.0**: Neural representations **strongly aligned** with semantic structure
  - Model has learned human-like categorical organization
  - Semantic categories are linearly separable in representation space
  
- **r = 0.3-0.7**: **Moderate alignment**
  - Model captures some semantic structure but not perfectly
  - May organize by other features (acoustics, visuals) in addition to semantics

- **r = 0.0-0.3**: **Weak/no alignment**
  - Neural organization differs substantially from semantic hierarchy
  - Model may use perceptual rather than conceptual features

### Expected patterns:

**Early layers (L0-L3):**
- Low semantic correlation (r < 0.3)
- Organized by low-level features (spectral patterns, edges)

**Middle layers (L4-L7):**
- Increasing semantic correlation (r = 0.3-0.6)
- Transition from perceptual to conceptual

**Late layers (L8-L11):**
- High semantic correlation (r > 0.6)
- Task-driven semantic organization
- **Fusion at L8** may show jump in correlation for combined RDMs

**Modality differences:**
- Audio may show higher semantic correlation (AudioSet is audio-centric)
- RGB may organize more by visual similarity
- Combined RDMs (post-fusion) should show highest correlation

---

## Conclusion

**The methodology is rigorous, well-designed, and follows established practices in representational similarity analysis.** 

The pipeline correctly:
1. ✅ Extracts meaningful neural representations (MLP outputs)
2. ✅ Computes appropriate distance metrics for each domain
3. ✅ Aligns classes correctly across different ID systems
4. ✅ Uses robust statistical comparison (Spearman correlation)
5. ✅ Provides interpretable visualizations

**No critical flaws identified.** The analysis is publication-ready with minor suggested enhancements being optional improvements rather than necessary corrections.
