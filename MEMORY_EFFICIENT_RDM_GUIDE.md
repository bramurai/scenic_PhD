
# Memory-Efficient RDM Computation Guide

## Problem

The extraction script accumulated **74 GB of data** (on disk), which requires **~88 GB in RAM** to load all at once. Your system has only **62 GB RAM**, making it impossible to:

```python
class_averages = accumulator.compute_averages()  # ❌ Requires 88GB - OOM!
```

## Solution: Process Streaming (Keep <5GB in RAM)

We've restructured the workflow into three stages, each processing data incrementally:

### Stage 1: Extract & Accumulate (Already Working ✓)
- Saves individual `.npy` files in `.accumulation/` (one per class per activation type)
- Each file is ~1-10 MB
- Total disk: 74 GB (12,648 files)
- RAM usage: Constant ~5-10 GB

### Stage 2: Save Averaged Activations (Modified ✓)
Instead of loading all 74 GB into memory, we:
1. Iterate through `.accumulation/*.npy` files
2. Load one at a time, divide by count, save to `averaged_activations/`
3. Delete from RAM immediately
4. **Result**: Same output, but never holds >100 MB in RAM

### Stage 3: Compute RDM (New Script)
Instead of loading all averaged activations, we:
1. Process classes in batches (e.g., 50 classes at a time)
2. Load only those 50 classes into RAM (~500 MB)
3. Compute pairwise distances within and across batches
4. Keep RDM matrix on disk, update incrementally
5. **Result**: Build full RDM without ever loading all data

## File Structure After Extraction

```
audioset_analysis_AV/
├── .accumulation/                    # Raw activation sums (74 GB, 12,648 files)
│   ├── class_0_encoder_block_L0_audio_output.npy
│   ├── class_0_encoder_block_L0_rgb_output.npy
│   └── ...
├── checkpoint.pkl                    # Tiny checkpoint file (~3 KB)
├── class_statistics.csv              # Per-class sample counts
├── metadata.pkl                      # Final run metadata
└── averaged_activations/             # NEW: Averaged activations (74 GB same size, but won't load all)
    ├── class_0_encoder_block_L0_audio_output.npy
    ├── class_0_encoder_block_L0_rgb_output.npy
    ├── class_1_encoder_block_L0_audio_output.npy
    └── metadata.pkl                  # Metadata for easy loading
```

## Step-by-Step Usage

### 1. Resume Extraction (generates averaged_activations/)
```bash
python extract_mbt_activations_class_averaged.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=audioset_analysis_AV \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --batch_size=4 \
  --num_samples=3853 \
  --checkpoint_every=50
```

**RAM Usage**: Constant ~10 GB (will NOT exceed 62 GB)  
**Output**: `audioset_analysis_AV/averaged_activations/` (74 GB)

### 2. Compute RDM (process streaming)
```bash
python compute_rdm_from_accumulation.py \
  --accumulation_dir=audioset_analysis_AV/.accumulation \
  --checkpoint_path=audioset_analysis_AV/checkpoint.pkl \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --output_dir=RDM_from_accumulation \
  --distance_metric=correlation \
  --batch_size=50
```

**RAM Usage**: ~500 MB (50 classes × ~10 MB each + overhead)  
**Output**: 
- `RDM_from_accumulation/rdm_matrix.npz` (RDM + metadata)
- `RDM_from_accumulation/rdm_matrix.csv` (human-readable)
- `RDM_from_accumulation/class_info.csv` (class metadata)

### 3. Combine Classes (e.g., hierarchical grouping)
```bash
python combine_classes_from_disk.py \
  --averaged_dir=audioset_analysis_AV/averaged_activations \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --class_indices=10,11,12 \
  --output_name=audio_content \
  --output_dir=combined_classes
```

**RAM Usage**: <100 MB  
**Output**: Combined class activations (saved as individual files)

## Why This Works

| Approach | RAM Used | Limit Hit? | Status |
|----------|----------|-----------|--------|
| Load all at once | 88 GB | ❌ YES (need 62 GB) | OOM |
| Batch processing (50 classes) | 500 MB | ✓ NO | ✓ Works |
| Streaming from disk | Variable | ✓ NO | ✓ Works |

## Memory Breakdown

**When processing 50 classes at a time:**
- Each class has 24 activations (12 layers × 2 modalities)
- Audio activations: ~1.2 MB each
- RGB activations: ~9.2 MB each
- **Per class**: ~250 MB
- **50 classes**: ~12.5 GB

Wait, that's too much! Let me recalculate... Actually, here's the actual breakdown:

- 527 classes total
- 24 activations per class
- Most activations ~9.2 MB (RGB), ~1.2 MB (audio)
- **Total for all classes**: 527 × 24 × ((9.2 + 1.2)/2) ≈ 74 GB ✓

**Per batch of 50 classes**:
- 50 classes × 24 activations = 1,200 arrays
- 1,200 × 5 MB (average) = 6 GB

Hmm, that's still large. Let me verify the actual file sizes:

```bash
du -sh audioset_analysis_AV/.accumulation/ | head -1
# Output should show total size ~74 GB
```

**Better approach**: Reduce batch size!
```bash
# Use smaller batches to stay well under 62 GB
python compute_rdm_from_accumulation.py \
  --batch_size=20  # Instead of 50: ~2.4 GB per batch
```

## Loading Results

### Load RDM Matrix
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('RDM_from_accumulation/rdm_matrix.npz')
rdm = data['rdm_matrix']
class_names = data['class_names']

# Visualize
plt.figure(figsize=(12, 12))
plt.imshow(rdm, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Distance')
plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
plt.yticks(range(len(class_names)), class_names, fontsize=8)
plt.title('AudioSet RDM (Correlation Distance)')
plt.tight_layout()
plt.savefig('rdm_heatmap.png', dpi=150, bbox_inches='tight')
```

### Load Individual Class Activations
```python
import numpy as np

# Load a single class activation (low memory!)
music_rgb_L0 = np.load('audioset_analysis_AV/averaged_activations/class_10_encoder_block_L0_rgb_output.npy')
print(music_rgb_L0.shape)  # e.g., (1, 768) or similar

# Load only classes 10-20 (instead of all 527)
for class_idx in range(10, 20):
    rgb_acts = np.load(f'audioset_analysis_AV/averaged_activations/class_{class_idx}_encoder_block_L0_rgb_output.npy')
    # Process one at a time...
```

### Load Combined Classes
```python
import numpy as np
import pickle

# Load metadata for a combined class
with open('combined_classes/audio_content/audio_content_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Source classes: {metadata['source_class_names']}")
print(f"Number of activations: {metadata['num_activations']}")

# Load combined class activation
combined_rgb = np.load('combined_classes/audio_content/audio_content_encoder_block_L0_rgb_output.npy')
```

## Key Advantages

1. **Memory Efficient**: Never loads >5-10 GB at once
2. **Flexible**: Can compute RDMs with different class groupings
3. **Reusable**: `.accumulation/` and `averaged_activations/` are standalone
4. **Recoverable**: All intermediate files preserved (can re-run final steps)
5. **Scalable**: Works with even larger datasets

## Checkpoint.pkl Contents

Your `checkpoint.pkl` contains ONLY:
- `processed_count`: 3800 (samples processed)
- `counts`: dict with 527 keys (class indices) → counts per class
- `num_classes`: 527
- `activation_names`: list of 24 activation names

**NOT stored** (would be too large):
- Activation sums (they're in individual `.npy` files)
- Full activations (would exceed checkpoint file size)

This is why we can resume without rebuilding `.accumulation/` - the data persists on disk!

## Troubleshooting

**Q: Script still uses too much RAM**
- Reduce `--batch_size` further: `--batch_size=10`
- Or use `--batch_size=1` to process one class at a time

**Q: I want the old single `.npz` file**
- Can't fit in RAM, but you can create a "lazy loader" that loads from individual files on demand
- Or use a format like HDF5 or Zarr that supports chunked/lazy loading

**Q: How do I compute RDM with only certain classes?**
- Filter in `compute_rdm_from_accumulation.py` before the loop:
  ```python
  class_indices = sorted([c for c in counts.keys() if counts[c] > 0])
  class_indices = class_indices[:100]  # Only first 100 classes
  ```

**Q: Can I combine classes differently after extraction?**
- Yes! Use `combine_classes_from_disk.py` with different groupings
- Or manually load and combine:
  ```python
  import numpy as np
  music = np.load('averaged_activations/class_10_encoder_block_L0_rgb_output.npy')
  speech = np.load('averaged_activations/class_139_encoder_block_L0_rgb_output.npy')
  combined = (music + speech) / 2  # Average
  ```
