"""Clean and validate VGGSound CSV for preprocessing.

This script:
1. Converts VGGSound CSV to the format expected by generate_audiovisual_from_file.py
2. Optionally validates which YouTube videos still exist
3. Generates train/test split CSVs

Usage:
    python clean_vggsound_csv.py \
        --input_csv=PreProcessing/vggsound.csv \
        --output_dir=PreProcessing \
        --validate_videos=False
"""

import os
import csv
from typing import Dict, List, Tuple
from absl import app
from absl import flags
from absl import logging
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_string('input_csv', None, 'Path to original VGGSound CSV.')
flags.DEFINE_string('output_dir', None, 'Directory for output CSVs.')
flags.DEFINE_string('video_dir', None, 'Directory where videos are stored (optional, for validation).')
flags.DEFINE_bool('validate_videos', False, 'Check if video files exist.')
flags.DEFINE_bool('split_by_original', True, 'Keep original train/test splits.')

flags.mark_flag_as_required('input_csv')
flags.mark_flag_as_required('output_dir')


def parse_vggsound_csv(input_path: str) -> pd.DataFrame:
    """Parse original VGGSound CSV format.
    
    Original format: video_id,start_time,label,split
    Example: --0PQM4-hqg,30,waterfall burbling,train
    
    Args:
        input_path: Path to original VGGSound CSV.
        
    Returns:
        DataFrame with parsed data.
    """
    logging.info(f"Reading {input_path}")
    
    # VGGSound CSV has no header
    df = pd.read_csv(
        input_path,
        names=['video_id', 'start_time', 'label', 'split'],
        dtype={'video_id': str, 'start_time': int, 'label': str, 'split': str}
    )
    
    logging.info(f"Loaded {len(df)} entries")
    logging.info(f"Train: {len(df[df['split']=='train'])}, Test: {len(df[df['split']=='test'])}")
    logging.info(f"Unique labels: {df['label'].nunique()}")
    
    return df


def convert_to_preprocessing_format(df: pd.DataFrame, video_dir: str = None) -> pd.DataFrame:
    """Convert to format expected by generate_audiovisual_from_file.py.
    
    Target format: video_path,start,end,label,clip_id
    
    Args:
        df: DataFrame with VGGSound data.
        video_dir: Optional directory where videos are stored.
        
    Returns:
        DataFrame in preprocessing format.
    """
    logging.info("Converting to preprocessing format")
    
    # VGGSound clips are 10 seconds long
    df['start'] = df['start_time']
    df['end'] = df['start_time'] + 10
    
    # Create video path
    if video_dir:
        df['video_path'] = df.apply(
            lambda row: os.path.join(video_dir, f"{row['video_id']}_{row['start_time']:06d}.mp4"),
            axis=1
        )
    else:
        # Relative path
        df['video_path'] = df.apply(
            lambda row: f"{row['video_id']}_{row['start_time']:06d}.mp4",
            axis=1
        )
    
    # Create clip_id
    df['clip_id'] = df.apply(
        lambda row: f"{row['video_id']}_{row['start_time']:06d}",
        axis=1
    )
    
    # Select and reorder columns
    output_df = df[['video_path', 'start', 'end', 'label', 'clip_id', 'split']].copy()
    
    return output_df


def validate_videos(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Check which video files exist.
    
    Args:
        df: DataFrame with video_path column.
        
    Returns:
        Tuple of (filtered_df with only existing videos, list of missing videos).
    """
    logging.info("Validating video files...")
    
    missing = []
    exists_mask = []
    
    for idx, row in df.iterrows():
        if os.path.exists(row['video_path']):
            exists_mask.append(True)
        else:
            exists_mask.append(False)
            missing.append(row['video_path'])
        
        if (idx + 1) % 1000 == 0:
            logging.info(f"Validated {idx + 1}/{len(df)} videos")
    
    filtered_df = df[exists_mask].copy()
    
    logging.info(f"Found {len(filtered_df)}/{len(df)} videos")
    logging.info(f"Missing {len(missing)} videos")
    
    return filtered_df, missing


def save_splits(df: pd.DataFrame, output_dir: str, split_by_original: bool = True):
    """Save train and test CSVs.
    
    Args:
        df: DataFrame with data.
        output_dir: Output directory.
        split_by_original: If True, use original train/test split.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if split_by_original and 'split' in df.columns:
        # Use original splits
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        # Remove split column before saving
        train_df = train_df.drop(columns=['split'])
        test_df = test_df.drop(columns=['split'])
    else:
        # Create 80/20 split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
    
    # Save CSVs
    train_path = os.path.join(output_dir, 'vggsound_train.csv')
    test_path = os.path.join(output_dir, 'vggsound_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"Saved train CSV: {train_path} ({len(train_df)} entries)")
    logging.info(f"Saved test CSV: {test_path} ({len(test_df)} entries)")
    
    # Also save combined CSV
    combined_path = os.path.join(output_dir, 'vggsound_combined.csv')
    combined_df = df.drop(columns=['split']) if 'split' in df.columns else df
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Saved combined CSV: {combined_path} ({len(combined_df)} entries)")


def save_missing_list(missing: List[str], output_dir: str):
    """Save list of missing videos.
    
    Args:
        missing: List of missing video paths.
        output_dir: Output directory.
    """
    if not missing:
        return
    
    missing_path = os.path.join(output_dir, 'vggsound_missing.txt')
    with open(missing_path, 'w') as f:
        for video in missing:
            f.write(f"{video}\n")
    
    logging.info(f"Saved missing videos list: {missing_path} ({len(missing)} entries)")


def create_label_mapping(df: pd.DataFrame, output_dir: str):
    """Create label to index mapping.
    
    Args:
        df: DataFrame with label column.
        output_dir: Output directory.
    """
    labels = sorted(df['label'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    mapping_path = os.path.join(output_dir, 'vggsound_labels.txt')
    with open(mapping_path, 'w') as f:
        for label, idx in label_to_idx.items():
            f.write(f"{idx},{label}\n")
    
    logging.info(f"Saved label mapping: {mapping_path} ({len(labels)} classes)")


def main(argv):
    del argv  # Unused.
    
    # Parse original CSV
    df = parse_vggsound_csv(FLAGS.input_csv)
    
    # Convert to preprocessing format
    df = convert_to_preprocessing_format(df, FLAGS.video_dir)
    
    # Validate videos if requested
    missing = []
    if FLAGS.validate_videos:
        if not FLAGS.video_dir:
            logging.warning("video_dir not specified, skipping validation")
        else:
            df, missing = validate_videos(df)
    
    # Save splits
    save_splits(df, FLAGS.output_dir, FLAGS.split_by_original)
    
    # Save missing videos list
    if missing:
        save_missing_list(missing, FLAGS.output_dir)
    
    # Save label mapping
    create_label_mapping(df, FLAGS.output_dir)
    
    logging.info("\n=== Summary ===")
    logging.info(f"Total entries: {len(df)}")
    if FLAGS.validate_videos and FLAGS.video_dir:
        logging.info(f"Valid videos: {len(df)}")
        logging.info(f"Missing videos: {len(missing)}")
    logging.info(f"Output directory: {FLAGS.output_dir}")
    logging.info("\nNext steps:")
    logging.info("1. Download missing videos (if any) using yt-dlp")
    logging.info("2. Run preprocessing:")
    logging.info(f"   python generate_audiovisual_from_file.py \\")
    logging.info(f"     --csv_path={os.path.join(FLAGS.output_dir, 'vggsound_train.csv')} \\")
    logging.info(f"     --output_path=/path/to/tfrecords/train \\")
    logging.info(f"     --decode_audio=True --num_shards=100")


if __name__ == '__main__':
    app.run(main)
