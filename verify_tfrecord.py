#!/usr/bin/env python3
"""Verify TFRecord contents and analyze if they match MBT paper specifications.

This script:
1. Loads a TFRecord and displays its contents
2. Shows RGB frames and audio spectrogram
3. Verifies they match expected specifications from the MBT paper

Usage:
    python verify_tfrecord.py --tfrecord_path=Datasets/audioset_eval/data-00000-of-00010.tfrecord --num_samples=3
"""

import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for cluster
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError as e:
    print(f"ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


def load_tfrecord_sample(tfrecord_path, sample_idx=0):
    """Load a single sample from TFRecord."""
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    for idx, raw_record in enumerate(dataset):
        if idx == sample_idx:
            example = tf.train.SequenceExample()
            example.ParseFromString(raw_record.numpy())
            return example
    
    return None


def extract_rgb_frames(example):
    """Extract RGB frames from SequenceExample."""
    if 'image/encoded' not in example.feature_lists.feature_list:
        return None
    
    frames = []
    for feature in example.feature_lists.feature_list['image/encoded'].feature:
        # Decode JPEG
        frame_encoded = feature.bytes_list.value[0]
        frame = tf.image.decode_jpeg(frame_encoded).numpy()
        frames.append(frame)
    
    return np.array(frames)


def extract_spectrogram(example):
    """Extract audio spectrogram from SequenceExample."""
    if 'WAVEFORM/feature/floats' not in example.feature_lists.feature_list:
        return None
    
    # Get chunk size (default 100)
    chunk_size = 100
    if 'WAVEFORM/chunk_size' in example.context.feature:
        chunk_size = example.context.feature['WAVEFORM/chunk_size'].int64_list.value[0]
    
    # Get n_mels (default 128)
    n_mels = 128
    if 'WAVEFORM/num_mel_bins' in example.context.feature:
        n_mels = example.context.feature['WAVEFORM/num_mel_bins'].int64_list.value[0]
    
    # Extract chunks
    chunks = []
    for feature in example.feature_lists.feature_list['WAVEFORM/feature/floats'].feature:
        chunk_flat = np.array(feature.float_list.value, dtype=np.float32)
        chunk = chunk_flat.reshape(chunk_size, n_mels)
        chunks.append(chunk)
    
    # Concatenate all chunks
    spectrogram = np.vstack(chunks)  # Shape: (time, freq)
    
    return spectrogram


def load_ground_truth_labels(eval_csv_path='Video_csvs/audioset_eval.csv', 
                              labels_csv_path='Video_csvs/audioset_labels.csv'):
    """Load ground truth labels from AudioSet eval CSV and map to label indices.
    
    Returns:
        dict: {clip_id: {'ytid': str, 'start': int, 'end': int, 'label_indices': list, 'label_names': list}}
    """
    try:
        import pandas as pd
        
        # Load eval CSV (has YTID, start_seconds, end_seconds, positive_labels)
        # Skip comment lines starting with #
        eval_df = pd.read_csv(eval_csv_path, sep=', ', skipinitialspace=True, 
                             comment='#', engine='python')
        
        # Load labels CSV (has index, display_name, description, etc.)
        labels_df = pd.read_csv(labels_csv_path)
        
        # Create mapping from mid (e.g., '/m/068hy') to index
        label_mid_to_idx = dict(zip(labels_df['mid'], labels_df['index']))
        label_mid_to_name = dict(zip(labels_df['mid'], labels_df['display_name']))
        
        ground_truth = {}
        
        for _, row in eval_df.iterrows():
            ytid = str(row.iloc[0]).strip()  # First column is YTID
            start = int(float(str(row.iloc[1]).strip()))  # Second column is start_seconds
            end = int(float(str(row.iloc[2]).strip()))  # Third column is end_seconds
            clip_id = f"{ytid}_{start}"
            
            # Parse positive labels (comma-separated with slashes, e.g., "/m/068hy,/m/07q6cd_")
            label_str = str(row.iloc[3]).strip() if len(row) > 3 else ""
            label_indices = []
            label_names = []
            
            if label_str and label_str != '':  # Skip empty entries
                # Remove outer quotes if present
                label_str = label_str.strip('"\'')
                
                # Split by comma
                label_mids = [l.strip() for l in label_str.split(',')]
                
                # Look up each label MID
                for label_mid in label_mids:
                    label_mid = label_mid.strip()
                    if label_mid in label_mid_to_idx:
                        idx = label_mid_to_idx[label_mid]
                        name = label_mid_to_name[label_mid]
                        label_indices.append(int(idx))
                        label_names.append(name)
            
            ground_truth[clip_id] = {
                'ytid': ytid,
                'start': start,
                'end': end,
                'label_indices': sorted(label_indices),
                'label_names': label_names
            }
        
        return ground_truth
    
    except Exception as e:
        print(f"WARNING: Could not load ground truth labels: {e}")
        import traceback
        traceback.print_exc()
        return {}


def extract_labels(example, label_csv_path='Video_csvs/audioset_labels.csv', 
                  ground_truth=None):
    """Extract and decode labels from SequenceExample, with ground truth comparison."""
    labels_info = {}
    
    # Get clip ID
    clip_id = None
    if 'clip/media_id' in example.context.feature:
        clip_id = example.context.feature['clip/media_id'].bytes_list.value[0].decode('utf-8')
        labels_info['clip_id'] = clip_id
    
    # Check for multi-hot labels (AudioSet)
    if 'clip/label/multi_hot' in example.context.feature:
        multi_hot = np.array(example.context.feature['clip/label/multi_hot'].float_list.value)
        active_indices = np.where(multi_hot > 0)[0].tolist()
        
        labels_info['type'] = 'multi-hot (AudioSet)'
        labels_info['active_indices'] = active_indices
        labels_info['num_active'] = len(active_indices)
        
        # Try to load label names
        if os.path.exists(label_csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(label_csv_path)
                labels_info['names'] = [df.loc[df['index'] == idx, 'display_name'].values[0] 
                                       for idx in active_indices if idx < len(df)]
            except:
                labels_info['names'] = []
        else:
            labels_info['names'] = []
        
        # Compare with ground truth
        if ground_truth and clip_id and clip_id in ground_truth:
            gt = ground_truth[clip_id]
            gt_indices = set(gt['label_indices'])
            pred_indices = set(active_indices)
            
            overlap = gt_indices & pred_indices
            missing = gt_indices - pred_indices
            extra = pred_indices - gt_indices
            
            labels_info['ground_truth_indices'] = gt['label_indices']
            labels_info['ground_truth_names'] = gt['label_names']
            labels_info['overlap_count'] = len(overlap)
            labels_info['missing_count'] = len(missing)
            labels_info['extra_count'] = len(extra)
            labels_info['match_percentage'] = 100.0 * len(overlap) / max(len(gt_indices), 1)
            
            if overlap == gt_indices == pred_indices:
                labels_info['label_match'] = '✓ PERFECT MATCH'
            elif len(overlap) == len(gt_indices):
                labels_info['label_match'] = '✓ All GT labels present (but extras exist)'
            elif len(overlap) > 0:
                labels_info['label_match'] = f'⚠️  Partial match: {len(overlap)}/{len(gt_indices)} GT labels found'
            else:
                labels_info['label_match'] = '✗ NO OVERLAP with ground truth'
    
    # Check for string label (single-label datasets)
    elif 'clip/label/string' in example.context.feature:
        label_string = example.context.feature['clip/label/string'].bytes_list.value[0].decode('utf-8')
        labels_info['type'] = 'string (single-label)'
        labels_info['label'] = label_string
    
    return labels_info


def analyze_spectrogram(spec):
    """Analyze spectrogram to verify it matches MBT paper specifications."""
    analysis = {}
    
    # Shape analysis
    time_frames, freq_bins = spec.shape
    analysis['shape'] = (time_frames, freq_bins)
    analysis['expected_shape'] = '(1000, 128) for 10s audio or (800, 128) for 8s audio'
    
    # Check if log-scale (dB) or linear
    # Log-scale spectrograms typically have negative values (dB)
    # Linear spectrograms are always >= 0
    min_val = spec.min()
    max_val = spec.max()
    mean_val = spec.mean()
    std_val = spec.std()
    
    analysis['min'] = min_val
    analysis['max'] = max_val
    analysis['mean'] = mean_val
    analysis['std'] = std_val
    
    # Determine scale type
    if min_val < 0:
        analysis['scale'] = 'LOG (dB) ✓ CORRECT'
        analysis['expected_range'] = 'Typically -80 to 0 dB'
    else:
        analysis['scale'] = 'LINEAR ✗ WRONG (should be log-scale per MBT paper)'
        analysis['expected_range'] = 'Linear power values (incorrect)'
    
    # Check frequency bins
    if freq_bins == 128:
        analysis['freq_bins'] = '128 ✓ CORRECT (matches MBT paper)'
    else:
        analysis['freq_bins'] = f'{freq_bins} ✗ WRONG (should be 128)'
    
    # Check time resolution (100 frames per second for 10ms hop)
    expected_fps = 100  # 10ms hop = 100 frames/sec
    duration_estimate = time_frames / expected_fps
    analysis['estimated_duration'] = f'{duration_estimate:.1f}s'
    
    if time_frames == 1000:
        analysis['time_frames'] = '1000 ✓ CORRECT for 10s audio'
    elif time_frames == 800:
        analysis['time_frames'] = '800 ✓ CORRECT for 8s audio'
    else:
        analysis['time_frames'] = f'{time_frames} (unusual, expected 800 or 1000)'
    
    return analysis


def visualize_sample(frames, spec, labels_info, analysis):
    """Visualize RGB frames and spectrogram with analysis."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title with label info
    title = f"TFRecord Sample Verification"
    if 'clip_id' in labels_info:
        title += f"\nClip ID: {labels_info['clip_id']}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Show 4 RGB frames
    num_frames_to_show = min(4, len(frames))
    for i in range(num_frames_to_show):
        ax = fig.add_subplot(gs[0, i])
        frame_idx = int(i * len(frames) / num_frames_to_show)
        ax.imshow(frames[frame_idx])
        ax.set_title(f'Frame {frame_idx}/{len(frames)}')
        ax.axis('off')
    
    # Show spectrogram
    ax_spec = fig.add_subplot(gs[1, :])
    im = ax_spec.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
    ax_spec.set_title('Audio Spectrogram (Frequency vs Time)', fontsize=12, fontweight='bold')
    ax_spec.set_xlabel('Time Frames')
    ax_spec.set_ylabel('Mel Frequency Bins')
    plt.colorbar(im, ax=ax_spec, label='Magnitude (dB)' if analysis['scale'].startswith('LOG') else 'Power')
    
    # Add analysis text
    ax_analysis = fig.add_subplot(gs[2:, :])
    ax_analysis.axis('off')
    
    analysis_text = "=== ANALYSIS ===\n\n"
    analysis_text += f"RGB Frames:\n"
    analysis_text += f"  • Total frames: {len(frames)}\n"
    analysis_text += f"  • Frame size: {frames[0].shape[0]}x{frames[0].shape[1]}\n"
    analysis_text += f"  • Expected: ~250 frames for 10s @ 25fps, ~75 for 3s @ 25fps\n\n"
    
    analysis_text += f"Audio Spectrogram:\n"
    analysis_text += f"  • Shape: {analysis['shape']} (time x freq)\n"
    analysis_text += f"  • Expected: {analysis['expected_shape']}\n"
    analysis_text += f"  • Frequency bins: {analysis['freq_bins']}\n"
    analysis_text += f"  • Time frames: {analysis['time_frames']}\n"
    analysis_text += f"  • Estimated duration: {analysis['estimated_duration']}\n\n"
    
    analysis_text += f"Spectrogram Values:\n"
    analysis_text += f"  • Scale: {analysis['scale']}\n"
    analysis_text += f"  • Range: [{analysis['min']:.2f}, {analysis['max']:.2f}]\n"
    analysis_text += f"  • Mean: {analysis['mean']:.2f}, Std: {analysis['std']:.2f}\n"
    analysis_text += f"  • Expected range: {analysis['expected_range']}\n\n"
    
    if analysis['scale'].startswith('LINEAR'):
        analysis_text += "⚠️  WARNING: Spectrogram is in LINEAR scale but MBT paper specifies LOG scale!\n"
        analysis_text += "   This will cause major distribution mismatch with pretrained model.\n\n"
    else:
        analysis_text += "✓  Spectrogram appears to be in LOG scale (correct)\n\n"
    
    # Add label comparison
    if 'ground_truth_names' in labels_info:
        analysis_text += "LABEL VERIFICATION:\n"
        analysis_text += f"  TFRecord labels: {', '.join(labels_info['names'][:5])}"
        if len(labels_info['names']) > 5:
            analysis_text += f" ... +{len(labels_info['names']) - 5}"
        analysis_text += f"\n  Ground Truth: {', '.join(labels_info['ground_truth_names'][:5])}"
        if len(labels_info['ground_truth_names']) > 5:
            analysis_text += f" ... +{len(labels_info['ground_truth_names']) - 5}"
        analysis_text += f"\n  {labels_info['label_match']}\n"
        analysis_text += f"  Match: {labels_info['match_percentage']:.1f}%"
    
    ax_analysis.text(0.05, 0.95, analysis_text, transform=ax_analysis.transAxes,
                    fontsize=9, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Verify TFRecord contents')
    parser.add_argument('--tfrecord_path', type=str, default=None,
                       help='Path to TFRecord file (if not provided, auto-detects first available)')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to verify')
    parser.add_argument('--output_dir', type=str, default='logs',
                       help='Directory to save verification plots')
    parser.add_argument('--label_csv', type=str, default='Video_csvs/audioset_labels.csv',
                       help='Path to AudioSet labels CSV')
    parser.add_argument('--eval_csv', type=str, default='Video_csvs/audioset_eval.csv',
                       help='Path to AudioSet eval CSV with ground truth labels')
    
    args = parser.parse_args()
    
    # Auto-detect TFRecord if not provided
    if args.tfrecord_path is None:
        tfrecords = sorted(glob.glob('Datasets/audioset_eval/**/*.tfrecord', recursive=True))
        if not tfrecords:
            print("ERROR: No TFRecords found in Datasets/audioset_eval/")
            sys.exit(1)
        args.tfrecord_path = tfrecords[0]
        print(f"Auto-detected TFRecord: {args.tfrecord_path}")
    
    if not os.path.exists(args.tfrecord_path):
        print(f"ERROR: TFRecord not found: {args.tfrecord_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ground truth labels
    print("Loading ground truth labels from CSV...")
    ground_truth = load_ground_truth_labels(args.eval_csv, args.label_csv)
    print(f"Loaded ground truth for {len(ground_truth)} clips")
    
    print(f"\n{'='*80}")
    print(f"VERIFYING TFRECORD: {args.tfrecord_path}")
    print(f"{'='*80}\n")
    
    # Track statistics
    stats = {
        'total_samples': 0,
        'perfect_matches': 0,
        'partial_matches': 0,
        'no_overlap': 0,
        'not_in_ground_truth': 0,
        'avg_match_percentage': 0
    }
    match_percentages = []
    
    # Process samples
    for sample_idx in range(args.num_samples):
        print(f"\n--- Sample {sample_idx} ---")
        
        # Load sample
        example = load_tfrecord_sample(args.tfrecord_path, sample_idx)
        if example is None:
            print(f"Could not load sample {sample_idx}")
            break
        
        # Extract data
        frames = extract_rgb_frames(example)
        spec = extract_spectrogram(example)
        labels_info = extract_labels(example, args.label_csv, ground_truth)
        
        stats['total_samples'] += 1
        
        # Print basic info
        print(f"Clip ID: {labels_info.get('clip_id', 'N/A')}")
        print(f"Label type: {labels_info.get('type', 'N/A')}")
        if 'names' in labels_info:
            print(f"TFRecord labels ({labels_info['num_active']}): {', '.join(labels_info['names'][:5])}")
            if len(labels_info['names']) > 5:
                print(f"  ... and {len(labels_info['names']) - 5} more")
        elif 'label' in labels_info:
            print(f"Label: {labels_info['label']}")
        
        # Print ground truth comparison
        if 'ground_truth_names' in labels_info:
            print(f"\nGround Truth labels ({len(labels_info['ground_truth_names'])}): {', '.join(labels_info['ground_truth_names'][:5])}")
            if len(labels_info['ground_truth_names']) > 5:
                print(f"  ... and {len(labels_info['ground_truth_names']) - 5} more")
            print(f"\nLabel Comparison: {labels_info['label_match']}")
            print(f"  • Overlap: {labels_info['overlap_count']} labels")
            print(f"  • Missing: {labels_info['missing_count']} labels")
            print(f"  • Extra: {labels_info['extra_count']} labels")
            print(f"  • Match: {labels_info['match_percentage']:.1f}%")
            
            match_percentages.append(labels_info['match_percentage'])
            
            # Update statistics
            if labels_info['label_match'].startswith('✓ PERFECT'):
                stats['perfect_matches'] += 1
            elif '✓' in labels_info['label_match']:
                stats['partial_matches'] += 1
            elif '⚠️' in labels_info['label_match']:
                stats['partial_matches'] += 1
            else:
                stats['no_overlap'] += 1
        else:
            print("\n⚠️  NOT FOUND in ground truth CSV")
            stats['not_in_ground_truth'] += 1
        
        if frames is not None:
            print(f"\nRGB frames: {len(frames)} frames, shape {frames[0].shape}")
        else:
            print("RGB frames: NOT FOUND")
        
        if spec is not None:
            print(f"Spectrogram: shape {spec.shape}")
            analysis = analyze_spectrogram(spec)
            print(f"  Scale: {analysis['scale']}")
            print(f"  Range: [{analysis['min']:.2f}, {analysis['max']:.2f}]")
            print(f"  Mean: {analysis['mean']:.2f}, Std: {analysis['std']:.2f}")
        else:
            print("Spectrogram: NOT FOUND")
            analysis = None
        
        # Visualize
        if frames is not None and spec is not None:
            fig = visualize_sample(frames, spec, labels_info, analysis)
            output_path = os.path.join(args.output_dir, f'tfrecord_verify_sample{sample_idx}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
            plt.close()
    
    # Print summary statistics
    if match_percentages:
        stats['avg_match_percentage'] = np.mean(match_percentages)
    
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples verified: {stats['total_samples']}")
    print(f"Perfect matches: {stats['perfect_matches']}")
    print(f"Partial matches: {stats['partial_matches']}")
    print(f"No overlap: {stats['no_overlap']}")
    print(f"Not in ground truth: {stats['not_in_ground_truth']}")
    if match_percentages:
        print(f"Average match percentage: {stats['avg_match_percentage']:.1f}%")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
