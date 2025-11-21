#!/usr/bin/env python3
"""Create correct AudioSet label mapping for your CSV.

This maps your CSV labels to AudioSet's official 527-class indices.
"""

import csv
import urllib.request

# AudioSet ontology CSV URL
ONTOLOGY_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

print("Step 1: Downloading official AudioSet class labels...")
with urllib.request.urlopen(ONTOLOGY_URL) as response:
    content = response.read().decode('utf-8')

# Parse AudioSet ontology
lines = content.strip().split('\n')
reader = csv.reader(lines)
header = next(reader)

# Create mapping: display_name -> official_index
audioset_name_to_index = {}
for row in reader:
    if len(row) >= 3:
        index = int(row[0])
        display_name = row[2]
        audioset_name_to_index[display_name] = index

print(f"Loaded {len(audioset_name_to_index)} AudioSet classes\n")

# Read your CSV labels
print("Step 2: Reading your CSV labels...")
csv_labels = set()
with open('Video_csvs/audioset_eval_100.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_labels.add(row['label'])

csv_labels = sorted(csv_labels)
print(f"Found {len(csv_labels)} unique labels in your CSV\n")

# Map your labels to AudioSet indices
print("Step 3: Mapping your labels to AudioSet indices...")
print("="*80)

mapping = {}
not_found = []

for label in csv_labels:
    if label in audioset_name_to_index:
        audioset_idx = audioset_name_to_index[label]
        mapping[label] = audioset_idx
        print(f"✓ {label:40s} -> AudioSet index {audioset_idx:3d}")
    else:
        not_found.append(label)
        print(f"✗ {label:40s} -> NOT FOUND in AudioSet!")

if not_found:
    print(f"\n⚠️  WARNING: {len(not_found)} labels not found in AudioSet ontology:")
    for label in not_found:
        print(f"  - {label}")
    print("\nYou may need to manually map these or find close matches.")

# Create corrected label_mapping.txt for your dataset
print("\nStep 4: Creating corrected label_mapping.txt...")
output_file = "Datasets/audioset_eval_100/label_mapping_corrected.txt"

with open(output_file, 'w') as f:
    for label in sorted(mapping.keys()):
        audioset_idx = mapping[label]
        f.write(f"{audioset_idx}\t{label}\n")

print(f"✓ Created: {output_file}")
print(f"\nThis file maps your {len(mapping)} CSV labels to AudioSet's official indices.")
print("\nNext steps:")
print("1. Replace Datasets/audioset_eval_100/label_mapping.txt with label_mapping_corrected.txt")
print("2. Re-run preprocessing with the corrected mapping")
print("3. Or re-extract activations if TFRecords are already created with wrong labels")
