#!/usr/bin/env python3
"""Convert audioset_eval.csv to audioset_ready.csv format for preprocessing.

This script transforms the AudioSet evaluation CSV into the format expected by
the preprocessing pipeline:
- YTID → video_path (with .mp4 extension and _000001 suffix)
- start_seconds → start (unchanged)
- end_seconds → end (adjusted to create 8-second clips)
- positive_labels → label (unchanged, will be mapped using audioset_labels.csv)
"""

import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string('input_csv', 'Video_csvs/audioset_eval.csv', 
                   'Path to input audioset_eval.csv')
flags.DEFINE_string('output_csv', 'Video_csvs/audioset_ready.csv',
                   'Path to output audioset_ready.csv')
flags.DEFINE_float('clip_duration', 8.0,
                  'Duration of clips in seconds (default: 8.0 for MBT AudioSet)')


def main(argv):
    del argv
    
    logging.info(f'Reading AudioSet eval CSV from {FLAGS.input_csv}')
    # AudioSet CSV has quoted labels with commas, need to handle properly
    df = pd.read_csv(FLAGS.input_csv, 
                     names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                     skiprows=3,  # Skip the header comments
                     skipinitialspace=True,
                     quotechar='"')
    
    logging.info(f'Found {len(df)} videos')
    
    # Transform columns
    # YTID → video_path: Add .mp4 extension and _000001 suffix
    df['video_path'] = df['YTID'] + '_000001.mp4'
    
    # start_seconds → start (keep as is)
    df['start'] = df['start_seconds']
    
    # end_seconds → end (adjust to start + clip_duration for 8-second clips)
    df['end'] = df['start'] + FLAGS.clip_duration
    
    # positive_labels → label (keep as is, will be mapped during preprocessing)
    df['label'] = df['positive_labels']
    
    # Create clip_id (same as video filename without extension)
    df['clip_id'] = df['YTID'] + '_000001'
    
    # Select only the columns we need
    output_df = df[['video_path', 'start', 'end', 'label', 'clip_id']]
    
    # Save to CSV
    logging.info(f'Saving to {FLAGS.output_csv}')
    output_df.to_csv(FLAGS.output_csv, index=False)
    
    logging.info(f'Successfully created {FLAGS.output_csv}')
    logging.info(f'Total examples: {len(output_df)}')
    logging.info(f'Clip duration: {FLAGS.clip_duration} seconds')
    
    # Show first few examples
    logging.info('\nFirst 5 examples:')
    print(output_df.head())
    
    # Show label distribution
    logging.info('\nLabel field preview (first 5 unique):')
    unique_labels = df['positive_labels'].unique()[:5]
    for label in unique_labels:
        print(f'  {label}')


if __name__ == '__main__':
    app.run(main)
