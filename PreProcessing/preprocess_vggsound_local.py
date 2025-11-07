#!/usr/bin/env python3
"""
Local VGGSound preprocessing script.
Processes videos from a local directory into TFRecords.
"""

import argparse
import os
import sys
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Add scenic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def process_single_video(video_info):
    """Process a single video and return the serialized SequenceExample."""
    try:
        from generate_audiovisual_from_file import create_sequence_example
        
        sequence_example = create_sequence_example(
            video_path=video_info['video_path'],
            start_time=0.0,
            end_time=10.0,
            label=video_info['label'],
            clip_id=video_info['video_id'],
            target_fps=25,
            decode_audio=True,
            audio_sample_rate=16000,
            n_mels=128,
            win_length_ms=25.0,
            hop_length_ms=10.0,
            min_resize=256
        )
        
        return (True, video_info['video_id'], sequence_example.SerializeToString())
    except Exception as e:
        return (False, video_info['video_id'], str(e))

def main():
    parser = argparse.ArgumentParser(description='Process VGGSound videos into TFRecords locally')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for TFRecords')
    parser.add_argument('--start_row', type=int, required=True, help='Start row in CSV')
    parser.add_argument('--end_row', type=int, required=True, help='End row in CSV')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (train/test)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    # Import the preprocessing module
    try:
        from generate_audiovisual_from_file import create_sequence_example
        import tensorflow as tf
    except ImportError as e:
        print(f"ERROR: Failed to import preprocessing modules: {e}")
        print("Please ensure generate_audiovisual_from_file.py is in the PreProcessing directory")
        sys.exit(1)
    
    # Read CSV rows
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    videos_to_process = []
    
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Process only the specified row range
        for idx in range(args.start_row, min(args.end_row, len(rows))):
            row = rows[idx]
            
            # Strip whitespace from all values (CSV has spaces in column names/values)
            row = {k.strip(): v.strip() for k, v in row.items()}
            
            # The video_path in CSV is the exact filename we need
            # e.g., "--0PQM4-hqg_000030.mp4"
            video_filename = row['video_path']
            video_file = video_dir / video_filename
            
            # Only process if video exists
            if video_file.exists():
                videos_to_process.append({
                    'video_id': row['clip_id'],  # Use clip_id as unique identifier
                    'video_path': str(video_file),
                    'label': row['label'],
                    'split': args.split
                })
    
    if not videos_to_process:
        print(f"No videos found in {args.video_dir} for rows {args.start_row}-{args.end_row}")
        sys.exit(0)
    
    print(f"Processing {len(videos_to_process)} videos from rows {args.start_row}-{args.end_row}")
    print(f"Using {args.num_workers} parallel workers")
    
    # Create TFRecord writers (2 shards per batch for parallel loading)
    num_shards = 2
    writers = []
    shard_paths = []
    
    for shard_idx in range(num_shards):
        shard_path = output_dir / f"data-{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
        shard_paths.append(shard_path)
        writers.append(tf.io.TFRecordWriter(str(shard_path)))
    
    # Process videos in parallel
    successful = 0
    failed = 0
    
    try:
        with Pool(processes=args.num_workers) as pool:
            # Process videos in parallel
            for idx, result in enumerate(pool.imap(process_single_video, videos_to_process)):
                success, video_id, data = result
                
                if success:
                    # Write to shard (round-robin)
                    shard_idx = idx % num_shards
                    writers[shard_idx].write(data)
                    successful += 1
                    
                    if (idx + 1) % 10 == 0:
                        print(f"  Processed {idx + 1}/{len(videos_to_process)} videos...")
                else:
                    print(f"  Warning: Failed to process {video_id}: {data}")
                    failed += 1
        
        print(f"✓ Successfully processed {successful}/{len(videos_to_process)} videos")
        if failed > 0:
            print(f"  ({failed} videos failed)")
            
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close all writers
        for writer in writers:
            writer.close()

if __name__ == '__main__':
    main()
