#!/usr/bin/env python3
"""
QUICKSTART: Memory-Efficient MBT Activation Extraction & RDM Computation

TL;DR: Your data is 74 GB (too big for 62 GB RAM). Solution: Process streaming!

═══════════════════════════════════════════════════════════════════════════════
CURRENT STATUS
═══════════════════════════════════════════════════════════════════════════════

✓ Checkpoint saved with 3,800 samples processed (98.6% complete)
✓ .accumulation/ contains all raw sums (~74 GB, 12,648 files)
✓ Ready to resume extraction script

═══════════════════════════════════════════════════════════════════════════════
PROBLEM: OOM WHEN LOADING ALL DATA
═══════════════════════════════════════════════════════════════════════════════

Old approach:
    class_averages = accumulator.compute_averages()  # Loads 74 GB → OOM!

Issue:
    - 527 classes × 24 activations = 12,648 arrays
    - ~5.85 MB per array (average)
    - 12,648 × 5.85 MB = 74 GB
    - 74 GB > 62 GB → Out of Memory!

Solution: Never load everything at once. Process incrementally!

═══════════════════════════════════════════════════════════════════════════════
UPDATED WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

STAGE 1: Extract Activations (RESUME)
─────────────────────────────────────

This resumes from sample 3,800 and finishes processing the last 53 samples.
Output: averaged_activations/ directory (same data as before, just organized differently)

Command:
    cd ~/Documents/Bram/scenic_PhD
    python extract_mbt_activations_class_averaged.py \
      --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
      --checkpoint_dir=CheckPoints/MBT_AV \
      --test_data_dir=Datasets/audioset_eval \
      --output_dir=audioset_analysis_AV \
      --audioset_labels_csv=Video_csvs/audioset_labels.csv \
      --batch_size=4 \
      --num_samples=3853 \
      --checkpoint_every=50

RAM Usage: Constant 5-10 GB ✓
Time Estimate: ~10 minutes for 53 samples
Output: audioset_analysis_AV/averaged_activations/


STAGE 2: Compute RDM (NEW)
──────────────────────────

Computes pairwise distances between all classes WITHOUT loading all data.
Uses batching: loads 20 classes at a time (~2 GB), computes distances, unloads.

Command:
    python compute_rdm_from_accumulation.py \
      --accumulation_dir=audioset_analysis_AV/.accumulation \
      --checkpoint_path=audioset_analysis_AV/checkpoint.pkl \
      --audioset_labels_csv=Video_csvs/audioset_labels.csv \
      --output_dir=RDM_from_accumulation \
      --distance_metric=correlation \
      --batch_size=20

RAM Usage: ~2-3 GB ✓
Time Estimate: ~2-3 hours for 527×527 RDM
Output: 
    - RDM_from_accumulation/rdm_matrix.npz (RDM + metadata)
    - RDM_from_accumulation/rdm_matrix.csv (human-readable)
    - RDM_from_accumulation/class_info.csv (class info)


STAGE 3: Combine Classes (OPTIONAL)
────────────────────────────────────

Create custom class groupings (e.g., "all music genres" → single class).
Process one at a time, never load all data.

Example: Combine Music (class 10) + Speech (class 139) + Singing (class 11):

    python combine_classes_from_disk.py \
      --averaged_dir=audioset_analysis_AV/averaged_activations \
      --audioset_labels_csv=Video_csvs/audioset_labels.csv \
      --class_indices=10,11,139 \
      --output_name=audio_content \
      --output_dir=combined_classes

RAM Usage: <100 MB ✓
Output: combined_classes/audio_content/*.npy


═══════════════════════════════════════════════════════════════════════════════
COMPLETE EXAMPLE: END-TO-END
═══════════════════════════════════════════════════════════════════════════════

# 1. Resume extraction (completes last 53 samples)
python extract_mbt_activations_class_averaged.py \
  --config=scenic/projects/mbt/configs/audioset/Inference_config.py \
  --checkpoint_dir=CheckPoints/MBT_AV \
  --test_data_dir=Datasets/audioset_eval \
  --output_dir=audioset_analysis_AV \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --batch_size=4 --checkpoint_every=50

# 2. Compute RDM for all 527 classes
python compute_rdm_from_accumulation.py \
  --accumulation_dir=audioset_analysis_AV/.accumulation \
  --checkpoint_path=audioset_analysis_AV/checkpoint.pkl \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --output_dir=RDM_from_accumulation \
  --distance_metric=correlation \
  --batch_size=20

# 3. Create visualization
python << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

data = np.load('RDM_from_accumulation/rdm_matrix.npz')
rdm = data['rdm_matrix']

plt.figure(figsize=(12, 12))
plt.imshow(rdm, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Correlation Distance')
plt.title(f'AudioSet Class RDM ({rdm.shape[0]} classes)')
plt.tight_layout()
plt.savefig('rdm_heatmap.png', dpi=150)
print(f"Saved: rdm_heatmap.png")
EOF

═══════════════════════════════════════════════════════════════════════════════
CHECKPOINT.PKL CONTENTS (SMALL!)
═══════════════════════════════════════════════════════════════════════════════

Your checkpoint.pkl contains only:
{
    'processed_count': 3800,           # Samples processed so far
    'counts': {0: 245, 1: 198, ...},   # Counts per class
    'num_classes': 527,                # Total classes
    'activation_names': [              # 24 activation types
        'encoder_block_L0_audio_output',
        'encoder_block_L0_rgb_output',
        ...
    ]
}

Size: ~3 KB (not 19 GB like before!)

The actual activation data remains in:
- .accumulation/class_*.npy          # Raw sums (74 GB on disk)
- averaged_activations/class_*.npy   # Divided by counts (74 GB on disk)

═══════════════════════════════════════════════════════════════════════════════
FILE STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

audioset_analysis_AV/
├── .accumulation/                          # ← Raw accumulated sums (74 GB)
│   ├── class_0_encoder_block_L0_audio_output.npy
│   ├── class_0_encoder_block_L0_rgb_output.npy
│   └── ... (12,648 files total)
│
├── averaged_activations/                   # ← NEW: Averaged activations (74 GB)
│   ├── class_0_encoder_block_L0_audio_output.npy   (sum / count)
│   ├── class_0_encoder_block_L0_rgb_output.npy
│   └── ... (12,648 files total)
│   └── metadata.pkl                        # For easy loading
│
├── checkpoint.pkl                          # ← Tiny checkpoint (3 KB)
├── class_statistics.csv                    # Per-class sample counts
└── metadata.pkl                            # Run metadata

RDM_from_accumulation/
├── rdm_matrix.npz                         # RDM + metadata (compressed)
├── rdm_matrix.csv                         # Human-readable
└── class_info.csv                         # Class indices, names, sample counts

═══════════════════════════════════════════════════════════════════════════════
ANSWER TO YOUR ORIGINAL QUESTION
═══════════════════════════════════════════════════════════════════════════════

Q: "What's in the checkpoint files?"
A: Very little! Only metadata:
   - processed_count (how far we got)
   - counts (class sample counts)
   - num_classes and activation_names

Q: "Could I combine classes with just .accumulation files?"
A: YES! 100%! You have everything you need:
   - All raw sums are in .accumulation/*.npy
   - For custom grouping:
     1. Load class_10_encoder_block_L0_audio_output.npy (raw sum)
     2. Load class_11_encoder_block_L0_audio_output.npy (raw sum)
     3. Add them together: combined_sum = sum_10 + sum_11
     4. Divide by combined count: avg = combined_sum / (count_10 + count_11)
   - Done! No need for the main .npz files

Q: "Why did extraction crash?"
A: When trying to save the final averaged_activations.npz, the script tried
   to load ALL 74 GB into memory at once. Now it streams instead.

═══════════════════════════════════════════════════════════════════════════════
TESTING MEMORY USAGE
═══════════════════════════════════════════════════════════════════════════════

Monitor in real-time:
    watch -n 1 free -h

Before extraction:
    total        used        free
    62Gi        30Gi        32Gi

During extraction (should stay constant):
    62Gi        42Gi        20Gi    ← RAM stable!

During RDM computation with batch_size=20:
    62Gi        35Gi        27Gi    ← Stays well under limit!

═══════════════════════════════════════════════════════════════════════════════
IF YOU WANT TO COMBINE CLASSES RIGHT NOW
═══════════════════════════════════════════════════════════════════════════════

You don't even need to finish extraction! You can work with .accumulation/ now:

import numpy as np

# Load raw sums for two classes
music_sum = np.load('audioset_analysis_AV/.accumulation/class_10_encoder_block_L0_rgb_output.npy')
speech_sum = np.load('audioset_analysis_AV/.accumulation/class_139_encoder_block_L0_rgb_output.npy')

# Get counts from checkpoint
import pickle
with open('audioset_analysis_AV/checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

music_count = data['counts'][10]
speech_count = data['counts'][139]

# Combine
combined_sum = music_sum + speech_sum
combined_count = music_count + speech_count
combined_avg = combined_sum / combined_count

print(f"Combined average shape: {combined_avg.shape}")

═══════════════════════════════════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

1. ✓ Read this file (you're doing it!)
2. Read: MEMORY_EFFICIENT_RDM_GUIDE.md (full details)
3. Run Stage 1: Resume extraction (10 minutes)
4. Run Stage 2: Compute RDM (2-3 hours)
5. Create visualizations (minutes)

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == '__main__':
    print(__doc__)
