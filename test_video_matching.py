#!/usr/bin/env python3
"""
Test script to verify video matching logic between CSV and extracted videos.
"""

import csv
from pathlib import Path

# Configuration
csv_file = "PreProcessing/vggsound_train.csv"
video_dir = Path("vggsound_temp/videos_10")  # Update to match your actual directory

print("=" * 60)
print("Video Matching Test")
print("=" * 60)
print(f"CSV file: {csv_file}")
print(f"Video directory: {video_dir}")
print()

# Check if video directory exists
if not video_dir.exists():
    print(f"ERROR: Video directory not found: {video_dir}")
    print("Please extract videos first or check the path.")
    exit(1)

# Get list of actual video files
actual_videos = set(f.name for f in video_dir.glob("*.mp4"))
print(f"Found {len(actual_videos)} video files in directory")
print()

# Sample of actual video filenames
print("Sample of actual video filenames:")
for i, video in enumerate(sorted(actual_videos)[:5]):
    print(f"  {video}")
print()

# Parse CSV and check matching
found = []
missing = []
checked = 0
sample_csv_names = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    
    # Check first 50 rows as a test
    for idx, row in enumerate(reader):
        if idx >= 50:
            break
        
        # Strip whitespace from all values (CSV has spaces)
        row = {k.strip(): v.strip() for k, v in row.items() if k and v}
        
        # The video_path in CSV is the exact filename we need
        video_filename = row.get('video_path', '')
        
        if not video_filename:
            continue
        
        checked += 1
        if checked <= 5:
            sample_csv_names.append(video_filename)
        
        if video_filename in actual_videos:
            found.append(video_filename)
        else:
            missing.append(video_filename)

print("Sample of CSV video_path entries:")
for name in sample_csv_names:
    print(f"  {name}")
print()

print("=" * 60)
print("Results:")
print("=" * 60)
print(f"Checked: {checked} CSV entries")
print(f"Found: {len(found)} videos")
print(f"Missing: {len(missing)} videos")
print(f"Match rate: {len(found)/checked*100:.1f}%")
print()

if missing:
    print(f"First 5 missing videos:")
    for video in missing[:5]:
        print(f"  {video}")
    print()

if found:
    print(f"First 5 found videos:")
    for video in found[:5]:
        print(f"  {video}")
    print()

# Check if the issue is with the entire CSV not matching
if len(found) == 0:
    print("⚠️  WARNING: No matches found!")
    print()
    print("This suggests a mismatch between CSV filenames and extracted videos.")
    print()
    print("Possible issues:")
    print("  1. CSV video_path format doesn't match extracted filename format")
    print("  2. Videos extracted from wrong tar file (tar 00 vs CSV entries)")
    print("  3. Extraction path issue (wrong --strip-components value)")
elif len(missing) > len(found):
    print("⚠️  WARNING: More missing than found!")
    print()
    print("This is expected if videos are split across multiple tar files.")
    print("Tar 00 only contains a subset of all videos in the CSV.")
else:
    print("✓ Video matching appears to be working correctly!")
    print()
    print("The CSV entries match the extracted video filenames.")

print()
print("=" * 60)
