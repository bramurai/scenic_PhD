#!/usr/bin/env python3
"""
Create a complete AudioSet label mapping file with all 527 classes.
Maps AudioSet indices (0-526) to class names.
"""

import json
import urllib.request
import sys

def download_audioset_ontology():
    """Download AudioSet ontology JSON."""
    url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
    
    print(f"Downloading AudioSet ontology from {url}...")
    
    response = urllib.request.urlopen(url)
    data = response.read().decode('utf-8')
    
    # Parse CSV
    lines = data.strip().split('\n')
    
    # Skip header
    header = lines[0]
    print(f"Header: {header}")
    
    label_mapping = {}
    
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 3:
            index = int(parts[0])
            mid = parts[1].strip()
            display_name = ','.join(parts[2:]).strip().strip('"')
            
            label_mapping[index] = display_name
    
    print(f"Loaded {len(label_mapping)} classes")
    return label_mapping


def save_label_mapping(label_mapping, output_path):
    """Save label mapping in the format: index<TAB>name"""
    
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'w') as f:
        for idx in sorted(label_mapping.keys()):
            f.write(f"{idx}\t{label_mapping[idx]}\n")
    
    print(f"Saved {len(label_mapping)} labels")
    
    # Show some examples
    print("\nExample mappings:")
    for idx in list(sorted(label_mapping.keys()))[:10]:
        print(f"  {idx}: {label_mapping[idx]}")


if __name__ == '__main__':
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'audioset_ontology.txt'
    
    label_mapping = download_audioset_ontology()
    save_label_mapping(label_mapping, output_path)
    
    print(f"\n✓ Complete! Use this file with compute_rdm.py:")
    print(f"  --label_mapping_file={output_path}")
