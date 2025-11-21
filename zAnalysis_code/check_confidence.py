#!/usr/bin/env python3
"""Check prediction confidence by looking at logit margins."""

import numpy as np
import glob

# Load all samples and check prediction confidence
sample_files = sorted(glob.glob('audioset_analysis_AV/sample_*.npz'))

# Load label mapping
label_to_name = {}
with open('audioset_analysis_AV/label_mapping.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            label_to_name[int(parts[0])] = parts[1]

print("PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)
print("Checking margin between top-1 and top-2 predictions...")
print("(Larger margin = more confident, Smaller margin = ambiguous)\n")

low_confidence = []

for sample_file in sample_files:
    data = np.load(sample_file)
    logits = data['logits']
    sample_idx = int(data['sample_idx'])
    
    if logits.ndim > 1:
        logits = logits[0]
    
    # Get top 2 predictions
    top2_indices = np.argsort(logits)[-2:][::-1]
    top1_class, top2_class = top2_indices
    top1_logit, top2_logit = logits[top2_indices]
    
    margin = top1_logit - top2_logit
    
    top1_name = label_to_name.get(top1_class, f'Class {top1_class}')
    top2_name = label_to_name.get(top2_class, f'Class {top2_class}')
    
    status = ""
    if margin < 0.3:
        status = " ⚠️ VERY AMBIGUOUS"
        low_confidence.append((margin, sample_idx, top1_name, top2_name))
    elif margin < 0.6:
        status = " ⚠️ Low confidence"
        low_confidence.append((margin, sample_idx, top1_name, top2_name))
    elif margin > 1.5:
        status = " ✓ High confidence"
    
    print(f"Sample {sample_idx:3d}: {top1_name:30s} ({top1_logit:6.3f}) "
          f"vs {top2_name:30s} ({top2_logit:6.3f}) | margin={margin:.3f}{status}")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(low_confidence)}/{len(sample_files)} samples have low confidence (margin < 0.6)")
print("=" * 80)

if low_confidence:
    print("\nMost ambiguous predictions (smallest margins):")
    low_confidence.sort()
    for margin, idx, top1, top2 in low_confidence[:15]:
        print(f"  Sample {idx:3d}: {margin:.3f} - {top1} vs {top2}")
