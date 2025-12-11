# Understanding Checkpoint vs Accumulation Files

## What's in the Checkpoint File

**File:** `audioset_analysis_AV/checkpoint.pkl`

```python
{
    'processed_count': 3800,           # How many samples have been processed
    'num_classes': 527,                # Total AudioSet classes
    'counts': {
        72: 147,                       # class_72 has 147 samples
        73: 82,                        # class_73 has 82 samples
        288: 63,                       # class_288 has 63 samples
        ...                            # 527 classes total
    },
    'activation_names': [
        'encoder_block_L0_audio_output',
        'encoder_block_L0_rgb_output',
        'encoder_block_L1_audio_output',
        ...                            # 24 layer types total
    ]
}
```

**Purpose:** Checkpoint is for **resuming extraction** if the script crashes

---

## What's in the Accumulation Directory

**Directory:** `audioset_analysis_AV/.accumulation/`

Contains **12,648 .npy files** (527 classes × 24 activation types)

Each file format: `class_{CLASS_IDX}_{ACTIVATION_NAME}.npy`

Example: `class_137_encoder_block_L0_rgb_output.npy`

```
Shape: (401, 768)     # 401 temporal positions × 768 embedding dims
Dtype: float32
Size: ~1.2 MB
Contents: SUM of all activations for class 137 at layer L0 RGB
```

**Key detail:** Each file contains the **SUM**, not the **average**
- To get average: `sum_array / num_samples_in_class`
- Number of samples available in checkpoint.pkl `counts` dict

---

## Can You Use ONLY the .accumulation Files?

**YES, absolutely!**

The checkpoint file is just convenience metadata. The actual data is entirely in `.accumulation/`

### Why checkpoint is optional:

1. **Checkpoint enables resume:** Skip to sample 3800 if crashed
2. **Accumulation files have all data:** Every activation sum is already saved
3. **Counts are in checkpoint:** But you can recompute from any source

---

## Example: Combine Classes for Custom RDM

You have all 527 individual class averages stored. To combine classes:

```python
import numpy as np

# Load activation sum for class 137 (Music)
class_137_sum = np.load('.accumulation/class_137_encoder_block_L0_rgb_output.npy')
class_137_count = 45  # From checkpoint

# Load activation sum for class 138 (Speech)
class_138_sum = np.load('.accumulation/class_138_encoder_block_L0_rgb_output.npy')
class_138_count = 89

# Combine into "Audio" super-class
combined_sum = class_137_sum + class_138_sum
combined_count = class_137_count + class_138_count
combined_avg = combined_sum / combined_count

# Now use combined_avg for RDM computation
```

---

## Practical Use Cases

### 1. Group by AudioSet Domain
```python
# Combine all music classes
music_classes = [137, 138, 139, ...]
combined_music = sum of all music class sums / total music samples

# Combine all speech classes  
speech_classes = [200, 201, 202, ...]
combined_speech = sum of all speech class sums / total speech samples

# Compute RDM between domains
rdm = distance(combined_music, combined_speech)
```

### 2. Custom Class Grouping
```python
# Group by instrument
instrument_groups = {
    'piano': [137, 140, 145],
    'guitar': [141, 142, 143],
    'violin': [148, 149],
}

# Combine each group
group_averages = {}
for group_name, class_indices in instrument_groups.items():
    total_sum = sum(load_class_sum(idx) for idx in class_indices)
    total_count = sum(class_counts[idx] for idx in class_indices)
    group_averages[group_name] = total_sum / total_count

# Compute RDM across instrument groups
```

### 3. Hierarchical RDMs
```python
# Level 1: 527 individual class RDMs
# Level 2: Domain-level RDMs (music, speech, environmental)
# Level 3: Super-domain RDMs (acoustic vs semantic)

# Use .accumulation files at each level
```

---

## File Organization

```
audioset_analysis_AV/
├── .accumulation/                    ← YOU NEED THIS
│   ├── class_0_encoder_block_L0_audio_output.npy
│   ├── class_0_encoder_block_L0_rgb_output.npy
│   ├── class_0_encoder_block_L1_audio_output.npy
│   ...
│   └── class_526_encoder_block_L11_rgb_output.npy
├── checkpoint.pkl                    ← Optional (for resume only)
├── class_averaged_activations.npz    ← Will be created when extract finishes
└── class_statistics.csv              ← Statistics per class
```

---

## Storage Breakdown

- **Total .npy files:** 12,648
- **Total size:** ~65 GB
- **Per file:** ~1.2 MB
- **Each contains:** float32 (401, 768) activation sum

---

## Key Insights

✓ Checkpoint file is **metadata only** (KB size)
- Just tracks `processed_count` for resume
- Not needed for post-processing

✓ Accumulation files have **all the data** (65 GB)
- Every activation sum for every class
- Sufficient to compute any RDM combination

✓ You can build custom groupings **without re-running extraction**
- Load any combination of class .npy files
- Combine by adding sums and dividing by total count
- Compute new RDMs with any distance metric

✓ Scalable design:
- Per-sample RDM (527 × 527)
- Domain-level RDM (6 × 6)
- Hierarchical RDMs at multiple levels
- All from the same .accumulation files

---

## Next Steps

1. **Finish extraction** - Re-run script to process final 53 samples
2. **Explore .accumulation/** - Use provided utility script to combine classes
3. **Compute custom RDMs** - Create domain-specific groupings
4. **Analyze hierarchically** - Build RDMs at multiple levels

See `use_accumulation_files.py` for working examples!
