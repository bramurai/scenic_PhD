#!/usr/bin/env python3
"""Download AudioSet class labels and create label_mapping.txt"""

import csv
import urllib.request

# AudioSet ontology CSV URL
ONTOLOGY_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

print("Downloading AudioSet class labels...")
print(f"From: {ONTOLOGY_URL}")

try:
    with urllib.request.urlopen(ONTOLOGY_URL) as response:
        content = response.read().decode('utf-8')
    
    # Parse CSV
    lines = content.strip().split('\n')
    reader = csv.reader(lines)
    
    # Skip header
    header = next(reader)
    print(f"Header: {header}")
    
    # Create label mapping
    label_mapping = {}
    for row in reader:
        if len(row) >= 3:
            index = int(row[0])
            mid = row[1]  # Machine ID
            display_name = row[2]
            label_mapping[index] = display_name
    
    # Save to file
    output_file = "audioset_analysis_AV/label_mapping.txt"
    with open(output_file, 'w') as f:
        for idx in sorted(label_mapping.keys()):
            f.write(f"{idx}\t{label_mapping[idx]}\n")
    
    print(f"\nCreated {output_file}")
    print(f"Total classes: {len(label_mapping)}")
    print("\nFirst 10 classes:")
    for idx in sorted(label_mapping.keys())[:10]:
        print(f"  {idx:3d}: {label_mapping[idx]}")
    
    print("\nSample classes from your data:")
    sample_indices = [66, 80, 102, 152, 159, 170, 180, 182, 228, 232]
    for idx in sample_indices:
        if idx in label_mapping:
            print(f"  {idx:3d}: {label_mapping[idx]}")
        else:
            print(f"  {idx:3d}: NOT FOUND")
            
except Exception as e:
    print(f"Error: {e}")
    print("\nFailed to download. Creating from known AudioSet structure...")
    print("You may need to manually create the label mapping.")
