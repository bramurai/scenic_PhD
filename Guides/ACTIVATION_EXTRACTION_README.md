# MBT Activation and Attention Weight Extraction

Complete guide for extracting neural activations and attention weights from trained MBT models for analysis (PCA, attention flow, etc.).

## What Gets Extracted

The extraction script (`extract_mbt_activations.py`) captures:

1. **24 Encoder Block Outputs** (final output of each encoder block)
   - 12 RGB encoder blocks (layers 0-11)
   - 12 Audio encoder blocks (layers 0-11)
   
2. **Bottleneck Tokens** (layers 8-11 only)
   - 5 bottleneck tokens per modality per layer
   - Captured from fused sequences
   
3. **Attention Weight Matrices** (token-to-token attention)
   - 24 attention matrices (12 RGB + 12 Audio)
   - Default: Averaged over 12 heads → shape `(seq_len, seq_len)`
   - Optional: Full per-head → shape `(1, 12, seq_len, seq_len)`

## Architecture Overview

```
MBT Model Structure (12 layers total):

Layers 0-7 (Pre-fusion):
  ┌─────────────────────┐         ┌─────────────────────┐
  │ RGB Encoder Block   │         │ Audio Encoder Block │
  │ - Self Attention    │         │ - Self Attention    │
  │ - MLP               │         │ - MLP               │
  │ (3137 tokens)       │         │ (401 tokens)        │
  └─────────────────────┘         └─────────────────────┘

Layer 8 (Fusion starts - bottlenecks injected):
  ┌─────────────────────┐         ┌─────────────────────┐
  │ RGB + 5 Bottlenecks │         │ Audio + 5 Bottleneck│
  │ (3142 tokens)       │         │ (406 tokens)        │
  └─────────────────────┘         └─────────────────────┘
         ↓                                   ↓
  Bottleneck tokens aggregate cross-modal information

Layers 8-11 (Fusion with bottlenecks):
  - Each modality attends to its tokens + bottlenecks
  - Bottlenecks average information from both modalities
```

## Quick Start

### Extract Activations (Recommended Settings)
```bash
 python extract_mbt_activations_class_averaged.py   --config=scenic/projects/mbt/configs/audioset/Inference_config.py   --checkpoint_dir=CheckPoints/MBT_AV   --test_data_dir=Datasets/audioset_eval   --output_dir=audioset_analysis_AV   --audioset_labels_csv=Video_csvs/audioset_labels.csv   --batch_size=4   --num_samples=3852   --checkpoint_every=1
# Extract everything (default):
python extract_mbt_activations_class_averaged.py \
  --config=--config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=audioset_analysis_All_date \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --batch_size=4 \
  --num_samples=3852 \
  --checkpoint_every=1

# Extract only logits and compute mAP (no activations):
python extract_mbt_activations_class_averaged.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=audioset_analysis_logits-mAP_date \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --batch_size=4 \
  --num_samples=3852 \
  --checkpoint_every=1 \
  --nosave_activations \
  --save_logits \
  --compute_map

# Extract only activations (no logits or mAP):
python extract_mbt_activations_class_averaged.py \
  --config=... \
  --save_activations \
  --nosave_logits \
  --nocompute_map
```

```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=/home/labuta/Documents/Bram/scenic_PhD/audioset_analysis_AV \
  --average_attention_heads=True \
  --resume=True \
  --clear_cache_every=8
```

**Output:** 9 files (~500 MB each) in `audioset_analysis_AV/`
- `sample_00000.npz` through `sample_00008.npz`
- `summary.npz` - All logits
- `config.pkl` - Configuration used
- `metadata.pkl` - Extraction metadata

**Output:** 9 files (~500 MB each) in `audioset_analysis_AV/`
- `sample_00000.npz` through `sample_00008.npz`
- `summary.npz` - All logits
- `config.pkl` - Configuration used
- `metadata.pkl` - Extraction metadata

### Load and Inspect Data

```python
import numpy as np

# Load a sample
data = np.load('audioset_analysis_AV/sample_00000.npz')

# Check what's available
print("Keys:", list(data.keys()))

# Encoder block outputs (24 total)
rgb_layer0 = data['encoder_block_L0_rgb_output']  # Shape: (1, 3137, 768)
audio_layer0 = data['encoder_block_L0_audio_output']  # Shape: (1, 401, 768)

# Bottleneck tokens (layers 8-11 only)
bottleneck_L8_rgb = data['bottleneck_L8_rgb']  # Shape: (1, 5, 768)
bottleneck_L8_audio = data['bottleneck_L8_audio']  # Shape: (1, 5, 768)

# Attention weights (token-to-token, averaged over heads)
attn_L0_rgb = data['attention_weights_L0_rgb']  # Shape: (3137, 3137)
attn_L0_audio = data['attention_weights_L0_audio']  # Shape: (401, 401)

# Fused layer attention (includes bottlenecks)
attn_L8_rgb = data['attention_weights_L8_rgb']  # Shape: (3142, 3142)
# Last 5 rows/cols are bottleneck attention

# Logits
logits = data['logits']  # Shape: (1, 527)
```

## What Each Output Contains

### Encoder Block Outputs

**Keys:** `encoder_block_L{0-11}_{rgb|audio}_output`

**Shapes:**
- RGB (layers 0-7): `(1, 3137, 768)` - 3136 patches + 1 CLS token
- Audio (layers 0-7): `(1, 401, 768)` - 400 patches + 1 CLS token  
- RGB (layers 8-11): `(1, 3142, 768)` - includes 5 bottleneck tokens
- Audio (layers 8-11): `(1, 406, 768)` - includes 5 bottleneck tokens

**What it represents:**
- Final output after MLP in each encoder block
- Hidden dimension: 768
- These are the representations you'd use for PCA, t-SNE, etc.

### Bottleneck Tokens

**Keys:** `bottleneck_L{8-11}_{rgb|audio}`

**Shape:** `(1, 5, 768)` - 5 bottleneck tokens, hidden dim 768

**What it represents:**
- Cross-modal information aggregators
- Present only in fusion layers (8-11)
- Average information from both RGB and audio streams
- The last 5 tokens in the fused sequences

### Attention Weight Matrices

**Keys:** `attention_weights_L{0-11}_{rgb|audio}`

**Shapes (with `--average_attention_heads=True`, default):**
- RGB (layers 0-7): `(3137, 3137)` - ~37.5 MB each
- Audio (layers 0-7): `(401, 401)` - ~0.6 MB each
- RGB (layers 8-11): `(3142, 3142)` - ~37.7 MB each (includes bottleneck attention)
- Audio (layers 8-11): `(406, 406)` - ~0.6 MB each (includes bottleneck attention)

**What it represents:**
- `attention[i, j]` = how much token `i` attends to token `j`
- Rows sum to 1.0 (normalized attention weights)
- Averaged over 12 attention heads for file size reduction

**Example - CLS token attention:**
```python
attn_L0_rgb = data['attention_weights_L0_rgb']
cls_attention = attn_L0_rgb[0, :]  # CLS token (row 0) attention to all tokens
top_5_tokens = np.argsort(cls_attention)[-5:]  # Which tokens does CLS attend to most?
```

**Example - Bottleneck attention (layers 8+):**
```python
attn_L8_rgb = data['attention_weights_L8_rgb']  # Shape: (3142, 3142)
bottleneck_attention = attn_L8_rgb[-5:, :]  # Last 5 rows = bottleneck tokens
# Which RGB tokens do bottlenecks attend to?
```

## File Size Comparison

| Configuration | Attention Shape | Size per Sample | Total (9 samples) |
|---------------|----------------|-----------------|-------------------|
| **Averaged (default)** | `(seq_len, seq_len)` | **~500 MB** | **~4.5 GB** |
| Full per-head | `(1, 12, seq_len, seq_len)` | ~5.5 GB | ~50 GB |

## Command-Line Options

## Command-Line Options

### Required Arguments

- `--config` - Path to config file (e.g., `scenic/projects/mbt/configs/audioset/Inference_config.py`)
- `--checkpoint_dir` - Directory with checkpoint files (e.g., `CheckPoints/MBT_AV`)
- `--test_data_dir` - Directory with test TFRecords (e.g., `Audioset_test`)

### Optional Arguments

- `--output_dir` - Output directory (default: `activation_analysis`)
- `--num_samples` - Number of samples to process (default: 100)
- `--checkpoint_step` - Specific checkpoint step to load (default: latest)
- `--average_attention_heads` - Average over attention heads (default: `True`)
  - `True` → shape `(seq_len, seq_len)`, 12x smaller files
  - `False` → shape `(1, 12, seq_len, seq_len)`, full per-head attention

### Examples

**Extract with averaged attention (recommended):**
```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Audioset_test \
  --output_dir=audioset_analysis_AV \
  --num_samples=9 \
  --average_attention_heads=True
```

**Extract full per-head attention (for detailed analysis):**
```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Audioset_test \
  --output_dir=audioset_analysis_full \
  --num_samples=9 \
  --average_attention_heads=False
```

## Attention Flow Analysis

### Understanding Attention Weights

The attention matrix shows how each token attends to other tokens:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('audioset_analysis_AV/sample_00000.npz')

# Get attention for layer 0 (RGB)
attn = data['attention_weights_L0_rgb']  # Shape: (3137, 3137)

# Visualize full attention matrix
plt.figure(figsize=(10, 10))
plt.imshow(attn, cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.title('Layer 0 RGB Attention Matrix')
plt.xlabel('Key Tokens')
plt.ylabel('Query Tokens')
plt.savefig('attention_L0_rgb.png', dpi=150)

# Analyze CLS token attention
cls_attn = attn[0, :]  # CLS is token 0
top_k = 10
top_indices = np.argsort(cls_attn)[-top_k:][::-1]
print(f"CLS token attends most to tokens: {top_indices}")
print(f"Attention weights: {cls_attn[top_indices]}")
```

### Tracking Attention Across Layers

```python
# Compare how CLS attention changes across layers
for layer in range(12):
    attn_rgb = data[f'attention_weights_L{layer}_rgb']
    cls_attn = attn_rgb[0, :]
    
    # Compute entropy (how focused vs distributed)
    entropy = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
    
    # Find max attention
    max_attn = np.max(cls_attn[1:])  # Exclude self-attention
    max_token = np.argmax(cls_attn[1:]) + 1
    
    print(f"Layer {layer}: entropy={entropy:.2f}, "
          f"max_attn={max_attn:.4f} to token {max_token}")
```

### Bottleneck Attention Analysis (Layers 8-11)

```python
# Analyze how bottleneck tokens attend to RGB/audio tokens
for layer in range(8, 12):
    attn_rgb = data[f'attention_weights_L{layer}_rgb']  # Shape: (3142, 3142)
    
    # Last 5 tokens are bottlenecks
    bottleneck_attn = attn_rgb[-5:, :]  # Shape: (5, 3142)
    
    # Average over bottleneck tokens
    avg_bottleneck_attn = np.mean(bottleneck_attn, axis=0)
    
    # Which RGB tokens do bottlenecks attend to most?
    top_rgb_tokens = np.argsort(avg_bottleneck_attn[:-5])[-10:]  # Exclude bottleneck tokens
    
    print(f"Layer {layer} bottlenecks attend most to RGB tokens: {top_rgb_tokens}")
```

### Cross-Modal Attention Flow

```python
# Compare RGB and audio attention patterns
layer = 8  # First fusion layer
attn_rgb = data[f'attention_weights_L{layer}_rgb']
attn_audio = data[f'attention_weights_L{layer}_audio']

# Extract bottleneck attention from both modalities
bn_rgb = attn_rgb[-5:, :-5]  # Bottleneck attention to RGB tokens (5 x 3137)
bn_audio = attn_audio[-5:, :-5]  # Bottleneck attention to audio tokens (5 x 401)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.imshow(bn_rgb, aspect='auto', cmap='viridis')
ax1.set_title('Bottleneck → RGB Attention')
ax1.set_ylabel('Bottleneck Tokens')
ax1.set_xlabel('RGB Tokens')

ax2.imshow(bn_audio, aspect='auto', cmap='viridis')
ax2.set_title('Bottleneck → Audio Attention')
ax2.set_ylabel('Bottleneck Tokens')
ax2.set_xlabel('Audio Tokens')

plt.tight_layout()
plt.savefig('bottleneck_attention_L8.png', dpi=150)
```

## PCA and Dimensionality Reduction

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load all encoder outputs for a layer
data = np.load('audioset_analysis_AV/sample_00000.npz')
layer = 11  # Final layer

# Get encoder outputs
rgb_output = data[f'encoder_block_L{layer}_rgb_output']  # (1, 3142, 768)
audio_output = data[f'encoder_block_L{layer}_audio_output']  # (1, 406, 768)

# Flatten batch dimension
rgb_tokens = rgb_output[0]  # (3142, 768)
audio_tokens = audio_output[0]  # (406, 768)

# Run PCA on RGB tokens
pca = PCA(n_components=50)
rgb_pca = pca.fit_transform(rgb_tokens)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title(f'PCA on Layer {layer} RGB Tokens')
plt.grid(True)
plt.savefig(f'pca_variance_L{layer}_rgb.png', dpi=150)

# Visualize first 2 PCs
plt.figure(figsize=(10, 8))
plt.scatter(rgb_pca[:, 0], rgb_pca[:, 1], alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title(f'Layer {layer} RGB Tokens - First 2 Principal Components')
plt.savefig(f'pca_2d_L{layer}_rgb.png', dpi=150)
```

## Comparing Activations Across Samples

```python
# Load multiple samples and compare
samples = []
for i in range(9):
    data = np.load(f'audioset_analysis_AV/sample_{i:05d}.npz')
    samples.append(data)

# Compare CLS token activations across samples
layer = 0
cls_activations = []

for sample_data in samples:
    rgb_output = sample_data[f'encoder_block_L{layer}_rgb_output']
    cls_token = rgb_output[0, 0, :]  # CLS is token 0, shape (768,)
    cls_activations.append(cls_token)

cls_activations = np.array(cls_activations)  # Shape: (9, 768)

# Run PCA on CLS tokens across samples
pca = PCA(n_components=2)
cls_pca = pca.fit_transform(cls_activations)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(cls_pca[:, 0], cls_pca[:, 1])
for i, (x, y) in enumerate(cls_pca):
    plt.annotate(f'Sample {i}', (x, y), fontsize=10)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title(f'Layer {layer} CLS Tokens Across 9 Samples')
plt.grid(True)
plt.savefig('cls_tokens_across_samples.png', dpi=150)
```

plt.savefig('cls_tokens_across_samples.png', dpi=150)
```

## Troubleshooting

### Issue: "Checkpoint not found"
- Ensure `--checkpoint_dir` points to your checkpoint directory
- Should contain checkpoint files (not the parent CheckPoints folder)
- Example: `CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec` not just `CheckPoints/MBT_AV`

### Issue: "Out of memory"
- Reduce `--num_samples` to process fewer samples at a time
- Attention matrices for RGB are large (~37.5 MB each with averaging)
- Consider using `--average_attention_heads=True` (default) for smaller files

### Issue: "No activations captured"
- Check that your model has the sow() calls in EncoderBlock
- Flax's intermediate capture should work automatically
- Log output will show which activations were captured

### Issue: "TFRecord parsing errors"
- Verify your config matches the TFRecord format
- Check `num_spec_frames`, `spec_shape`, `num_frames` in config
- Ensure TFRecords were generated with correct `--clip_duration` and `--rgb_duration`

### Issue: "Files are too large"
- Default settings with averaged attention: ~500 MB per sample
- Without averaging (`--average_attention_heads=False`): ~5.5 GB per sample
- Consider saving to external drive for large datasets

## Summary of Key Points

### What You Get

✅ **24 encoder block outputs** - Final representations from each layer (12 RGB + 12 audio)  
✅ **Bottleneck tokens** - Cross-modal aggregators from fusion layers (8-11)  
✅ **Attention weights** - Token-to-token attention matrices (averaged over heads by default)  
✅ **Logits** - Model predictions for each sample

### File Sizes

- **With averaging (default):** ~500 MB per sample, ~4.5 GB for 9 samples
- **Without averaging:** ~5.5 GB per sample, ~50 GB for 9 samples

### Best Practices

1. **Use averaged attention** (`--average_attention_heads=True`) for attention flow analysis
2. **Save to external drive** if processing many samples
3. **Check first sample** to verify extraction before processing all samples
4. **Use encoder block outputs** for PCA, t-SNE, and representation analysis
5. **Use attention matrices** for attention flow, bottleneck analysis, and visualization

## References

- **Flax Documentation:** https://flax.readthedocs.io/en/latest/
- **JAX Documentation:** https://jax.readthedocs.io/
- **MBT Paper:** Nagrani et al., "Attention Bottlenecks for Multimodal Fusion" (NeurIPS 2021)
- **Scenic Framework:** https://github.com/google-research/scenic
