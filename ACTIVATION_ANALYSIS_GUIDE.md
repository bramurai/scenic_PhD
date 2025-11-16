# MBT Activation Extraction and Analysis

This directory contains scripts to extract neural activations and attention weights from your trained MBT model for downstream analyses like PCA and attention flow visualization.

## Overview

**Scripts to use:**
- ✅ `extract_mbt_activations.py`: Working extraction script (use this!)
- ✅ `analyze_activations.py`: PCA and attention flow analysis

**Old/incomplete scripts (don't use):**
- ❌ `Inference_config.py`: Points to ViT checkpoint, not your trained MBT model
- ❌ Any other activation extraction scripts in the repo

## Usage

### Step 0: Find Your Checkpoints

First, see what checkpoints you have available:

```bash
# List all checkpoint directories
ls -la CheckPoints/

# See checkpoints in a specific run
ls -la CheckPoints/mbt_run1/

# Find the latest checkpoint step
cat CheckPoints/mbt_run1/checkpoint

# List all checkpoint steps
ls CheckPoints/mbt_run1/checkpoint_*
```

Checkpoint directory structure:
```
CheckPoints/
  mbt_run1/           ← Use this as --checkpoint_dir
    checkpoint        ← Points to latest checkpoint
    checkpoint_1      ← Step 1 checkpoint
    checkpoint_1000   ← Step 1000 checkpoint
    checkpoint_5000   ← Step 5000 checkpoint
    ...
  mbt_run2/           ← Different training run
    checkpoint
    checkpoint_1
    ...
```

### Step 1: Extract Activations from Trained MBT Model

**Example 1: Using a specific checkpoint directory**
```bash
conda activate scenic_phd

# Point to a specific trained model directory
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Audioset_test \
  --output_dir=audioset_analysis_AV \
  --num_samples=9
```

**Example 2: Using a specific checkpoint step**
```bash
# Load checkpoint at step 5000
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=/media/labuta/7f1ad7d2-a1d3-4a1f-ae81-7cb5dd2661a3/VGG_Preprocessed/test_tfrecords_local \
  --output_dir=activation_analysis_run2_step5000 \
  --num_samples=100
```

**Example 3: Compare multiple checkpoints**
```bash
# Extract from different runs to compare
for run in mbt_run1 mbt_run2 mbt_run3; do
  python extract_mbt_activations.py \
    --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
    --checkpoint_dir=CheckPoints/${run} \
    --test_data_dir=/media/labuta/7f1ad7d2-a1d3-4a1f-ae81-7cb5dd2661a3/VGG_Preprocessed/test_tfrecords_local \
    --output_dir=activation_analysis_${run} \
    --num_samples=100
done
```

**Parameters:**
- `--config`: Path to your training config file
- `--checkpoint_dir`: **Directory containing checkpoint files** (e.g., `CheckPoints/mbt_run1`)
  - This should point to the specific run folder, NOT the parent `CheckPoints/` folder
  - The directory should contain files like `checkpoint`, `checkpoint_1`, `checkpoint_2`, etc.
- `--checkpoint_step`: (Optional) Specific checkpoint step to load (e.g., 1000, 5000)
  - If not specified, loads the latest checkpoint
  - Useful for comparing model performance at different training stages
- `--test_data_dir`: Directory with test TFRecords
- `--output_dir`: Where to save extracted activations
- `--num_samples`: Number of samples to process

**Important:** This loads your **trained MBT model checkpoint**, NOT the pretrained ViT. It only does forward passes, no training.

**Output:**
```
activation_analysis/
  sample_00000.npz  # Activations for sample 0
  sample_00001.npz  # Activations for sample 1
  ...
  summary.npz       # All logits
  metadata.pkl      # Config and metadata
  config.pkl        # Full configuration
```

Each `sample_XXXXX.npz` contains:
- `logits`: Model predictions
- `activation_<layer_name>`: Activations from each layer
- `attention_<layer_name>`: Attention weights from each attention layer

### Step 2: Analyze Activations (PCA, Attention Flow)

```bash
python analyze_activations.py \
  --activation_dir=activation_analysis \
  --output_dir=pca_results \
  --n_components=50 \
  --run_tsne=False
```

**Output:**
```
pca_results/
  pca_results.pkl          # PCA transformations for all layers
  attention_summary.pkl    # Attention weight statistics
  variance_explained.png   # Variance explained plots
  attention_maps.png       # Average attention visualizations
  tsne.png                 # (if --run_tsne=True)
```

### Step 3: Load Results in Python/Notebook

```python
import numpy as np
import pickle

# Load single sample activations
data = np.load('activation_analysis/sample_00000.npz')
logits = data['logits']
layer_activation = data['activation_Encoder_encoderblock_0']  # Example layer

# Load PCA results
with open('pca_results/pca_results.pkl', 'rb') as f:
    pca_results = pickle.load(f)

# Access PCA-transformed features
for layer_name, result in pca_results.items():
    print(f"{layer_name}:")
    print(f"  Shape: {result['transformed'].shape}")
    print(f"  Variance explained: {result['cumulative_variance'][:5]}")
    
    # Get first 10 principal components
    pc_10 = result['transformed'][:, :10]
    
    # Do your analysis...

# Load attention weights
with open('pca_results/attention_summary.pkl', 'rb') as f:
    attention = pickle.load(f)

for layer_name, summary in attention.items():
    avg_attn = summary['average_attention']
    # Analyze attention flow...
```

## What You Can Do with These Activations

### 1. PCA Analysis
- Dimensionality reduction of layer activations
- Find principal components of neural representations
- Analyze variance explained by components
- Compare representations across layers

### 2. Attention Flow Analysis
- Visualize which tokens attend to which
- Track attention patterns across layers
- Identify important features (e.g., audio vs visual tokens)
- Analyze cross-modal attention in bottleneck layers

### 3. Clustering
```python
from sklearn.cluster import KMeans

# Cluster samples based on layer activations
layer_features = pca_results['some_layer']['transformed'][:, :50]
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(layer_features)
```

### 4. Similarity Analysis
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity between samples
layer_features = pca_results['some_layer']['transformed']
similarity_matrix = cosine_similarity(layer_features)
```

### 5. Layer Comparison
```python
# Compare different layers
layer1_features = pca_results['layer1']['transformed']
layer2_features = pca_results['layer2']['transformed']

# Canonical Correlation Analysis, etc.
```

## Troubleshooting

### "No checkpoint found"
- Make sure `--checkpoint_dir` points to your trained MBT model directory
- Check if there's a `checkpoint` file in that directory
- Try specifying `--checkpoint_step` explicitly

### "No TFRecords found"
- Verify the path in `--test_data_dir`
- The script looks recursively for `*.tfrecord` files
- Make sure you've generated test TFRecords

### Memory issues
- Reduce `--num_samples`
- Process in batches
- Use `--n_components` with fewer components

### Attention weights not extracted
- Not all layers have attention weights
- Only transformer attention layers will have these
- Check the layer names in the output

## Notes

- The extraction script uses your **trained MBT checkpoint**, not the ViT pretrained model
- No training occurs - only forward passes
- Activations are saved per-sample for flexibility
- All intermediate layer outputs are captured
- Attention weights are extracted where available (transformer layers)

## Next Steps

After extraction, you can:
1. Run custom analyses on the saved `.npz` files
2. Create visualizations of attention patterns
3. Compare representations across modalities (audio vs video)
4. Analyze bottleneck token behavior
5. Study cross-modal fusion at different layers
