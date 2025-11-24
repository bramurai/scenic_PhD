# Class-Averaged Activation Extraction Guide

## Problem
Storing activations for all 3,853 AudioSet samples requires ~1.9 TB of storage (3,853 × 0.5 GB per sample).

## Solution
Compute class-averaged activations on-the-fly, handling multi-label data properly.

**Storage: ~26 GB** (527 classes × ~50 MB) - **100x smaller!**

## How It Works

### Multi-Label Handling
Each sample contributes to ALL its classes:
- Sample with labels [Music, Speech] → contributes to both Music AND Speech averages
- Average 2.56 labels per sample means each sample contributes to ~2.56 class averages

### What Gets Saved
For each of 527 classes:
- Encoder block outputs (24 layers: L0-L11 for RGB and Audio)
- Optional: Attention weights (if `--save_attention=True`)

## Usage

```bash
python extract_mbt_activations_class_averaged.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=audioset_analysis_AV \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv
```

### Optional Flags
- `--num_samples=N`: Process only first N samples (default: all 3,853)
- `--save_attention=True`: Also save attention weights (increases storage)
- `--clear_cache_every=50`: How often to clear JAX cache (lower = less memory)

## Output Files

### 1. `class_averaged_activations.npz` (~26 GB)
Main file containing averaged activations:

```python
import numpy as np

data = np.load('audioset_class_averaged/class_averaged_activations.npz')

# Get activation for Music (class 137), Layer 0, RGB modality
music_L0_rgb = data['class_137_encoder_block_L0_rgb_output']
# Shape: (seq_len, hidden_dim), e.g., (197, 768)

# Get activation for Speech (class 0), Layer 11, Audio modality  
speech_L11_audio = data['class_0_encoder_block_L11_audio_output']
# Shape: (seq_len, hidden_dim), e.g., (100, 768)

# Metadata
class_names = data['class_names']  # Array of 527 class names
samples_per_class = data['samples_per_class']  # How many samples per class
```

### 2. `class_statistics.csv`
Sample counts per class:

```csv
index,mid,display_name,num_samples
0,/m/09x0r,Speech,926
137,/m/04rlf,Music,1069
...
```

### 3. `metadata.pkl`
Processing metadata and configuration.

## Array Naming Convention

Format: `class_{index}_{activation_name}`

Examples:
- `class_137_encoder_block_L0_rgb_output` - Music, Layer 0, RGB
- `class_0_encoder_block_L11_audio_output` - Speech, Layer 11, Audio
- `class_137_attention_weights_L5_rgb` - Music, Layer 5, RGB attention (if saved)

## For RDM Analysis

Perfect for computing Representational Dissimilarity Matrices:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Load class-averaged activations
data = np.load('audioset_class_averaged/class_averaged_activations.npz')

# Extract layer 6 RGB activations for all classes
num_classes = 527
layer_activations = []

for class_idx in range(num_classes):
    key = f'class_{class_idx}_encoder_block_L6_rgb_output'
    if key in data:
        act = data[key]
        # Flatten to 1D: (seq_len, hidden_dim) -> (seq_len * hidden_dim,)
        layer_activations.append(act.flatten())

layer_activations = np.array(layer_activations)  # Shape: (num_classes, features)

# Compute RDM using correlation distance
rdm = squareform(pdist(layer_activations, metric='correlation'))
# Shape: (num_classes, num_classes)
```

## Storage Comparison

| Method | Storage | Use Case |
|--------|---------|----------|
| Per-sample | ~1.9 TB | Sample-level analysis, debugging |
| Class-averaged | ~26 GB | RDM analysis, class representations |
| Class-averaged + attention | ~50 GB | RDM + attention flow analysis |

## Expected Runtime

- ~3,853 samples at ~2 seconds/sample = ~2 hours
- Progress logged every 100 samples
- Resumable if crashes (though less critical with class averaging)

## Memory Requirements

- ~4-6 GB RAM (processes one sample at a time)
- JAX cache cleared every 50 samples by default
- Reduce `--clear_cache_every` if running out of memory

## Next Steps: Compute RDMs

After extraction, use the class-averaged activations to compute RDMs across layers:

```bash
python compute_rdm_from_class_averaged.py \
  --input_file=audioset_class_averaged/class_averaged_activations.npz \
  --output_dir=audioset_rdms \
  --distance_metric=correlation
```

This will create RDMs showing how different AudioSet classes are represented at each layer.
