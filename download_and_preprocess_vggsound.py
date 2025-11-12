"""Download VGGSound videos from YouTube on-the-fly during preprocessing.

This script downloads videos temporarily, processes them, and deletes them
to avoid storing ~200k videos permanently.

Usage:
  python download_and_preprocess_vggsound.py \
    --csv_path=PreProcessing/vggsound_train.csv \
    --output_path=/path/to/tfrecords \
    --temp_dir=/path/to/temp \
    --num_shards=100
"""

import os
import tempfile
import subprocess
from typing import Optional
from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
import numpy as np
import json

# We'll import the preprocessing functions later to avoid flag conflicts
gen_module = None

FLAGS = flags.FLAGS

flags.DEFINE_string('csv_path', None, 'Path to VGGSound CSV file.')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord files.')
flags.DEFINE_string('temp_dir', None, 'Temporary directory for downloads (optional).')
flags.DEFINE_integer('num_shards', 10, 'Number of output shards.')
flags.DEFINE_bool('decode_audio', True, 'Whether to decode audio spectrograms.')
flags.DEFINE_integer('target_fps', 25, 'Target frames per second.')
flags.DEFINE_integer('audio_sample_rate', 16000, 'Audio sample rate.')
flags.DEFINE_integer('n_mels', 128, 'Number of mel bins.')
flags.DEFINE_float('win_length_ms', 25.0, 'Window length in ms.')
flags.DEFINE_float('hop_length_ms', 10.0, 'Hop length in ms.')
flags.DEFINE_bool('skip_existing', True, 'Skip if TFRecord shard already processed.')
flags.DEFINE_integer('batch_size', 100, 'Process this many videos before cleaning temp files.')
flags.DEFINE_string('progress_file', None, 'File to track progress (auto-generated if not specified).')
flags.DEFINE_integer('save_progress_every', 50, 'Save progress every N videos.')
flags.DEFINE_bool('check_duration', True, 'Verify video duration before downloading segment.')
flags.DEFINE_string('cookies_from_browser', None, 'Browser to extract cookies from (chrome, firefox, edge, etc).')
flags.DEFINE_string('cookies_file', None, 'Path to cookies.txt file for yt-dlp authentication.')
flags.DEFINE_string('local_videos_dir', None, 'If set, treat video paths as relative to this directory and use local mp4 files instead of downloading from YouTube.')
flags.DEFINE_bool('require_local', False, 'If True and local_videos_dir is set, skip videos not found locally instead of falling back to YouTube download.')
flags.DEFINE_bool('local_are_clips', False, 'If True, local videos are already 10s clips (extract 0-10s instead of using CSV start/end times).')

flags.mark_flag_as_required('csv_path')
flags.mark_flag_as_required('output_path')


def get_video_duration(video_id: str) -> Optional[float]:
    """Get video duration in seconds using yt-dlp.
    
    Args:
        video_id: YouTube video ID.
        
    Returns:
        Duration in seconds, or None if unavailable.
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-warnings',
            '--print', 'duration',
            '--no-playlist',
            '--sleep-requests', '1',  # Sleep 1 second between requests to avoid rate limiting
        ]
        
        # Add cookie authentication if provided
        if FLAGS.cookies_from_browser:
            cmd.extend(['--cookies-from-browser', FLAGS.cookies_from_browser])
        elif FLAGS.cookies_file:
            cmd.extend(['--cookies', FLAGS.cookies_file])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, timeout=30, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return None
            
    except Exception:
        return None


def get_local_video_duration(file_path: str) -> Optional[float]:
    """Get duration of a local video file using ffprobe.

    Returns duration in seconds or None if it cannot be determined.
    """
    try:
        # ffprobe returns duration on a single line with -show_entries format=duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'format=duration',
            '-of', 'default=nw=1:nk=1',
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=20, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        return None
    except Exception:
        return None


def download_youtube_video(video_id: str, start_time: int, output_path: str, 
                          check_duration: bool = True) -> bool:
    """Download a 10-second clip from YouTube using yt-dlp.
    
    Args:
        video_id: YouTube video ID.
        start_time: Start time in seconds.
        output_path: Output file path.
        check_duration: Whether to verify video duration before downloading.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Check if the video is long enough for the requested segment
        if check_duration:
            duration = get_video_duration(video_id)
            if duration is None:
                logging.warning(f"Could not get duration for {video_id}, attempting download anyway")
            elif start_time + 10 > duration:
                logging.warning(f"Video {video_id} is only {duration:.1f}s long, "
                              f"cannot extract segment at {start_time}s-{start_time+10}s")
                return False
            # else: duration is sufficient, proceed with download
        
        # Download full video at lowest quality (faster than you'd think for "worst")
        # Let ffmpeg handle the segment extraction for better compatibility
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-warnings',
            '--format', 'worst[ext=mp4]/worst',  # Lowest quality, prefer mp4
            '--output', output_path,
            '--no-playlist',
            '--concurrent-fragments', '4',  # Download fragments in parallel
            '--throttled-rate', '100K',  # Skip if speed drops below 100KB/s
            '--sleep-requests', '1',  # Sleep 1 second between requests to avoid rate limiting
        ]
        
        # Add cookie authentication if provided
        if FLAGS.cookies_from_browser:
            cmd.extend(['--cookies-from-browser', FLAGS.cookies_from_browser])
        elif FLAGS.cookies_file:
            cmd.extend(['--cookies', FLAGS.cookies_file])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            # Verify downloaded file has reasonable size (not empty/corrupted)
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # Less than 1KB is suspicious
                logging.warning(f"Downloaded file for {video_id} is too small ({file_size} bytes)")
                return False
            return True
        else:
            # Log detailed error for debugging
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "Unknown error"
            logging.warning(f"Failed to download {video_id} at {start_time}s")
            logging.warning(f"yt-dlp error: {error_msg}")
            logging.warning(f"Return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout downloading {video_id} after 60 seconds")
        return False
    except Exception as e:
        logging.warning(f"Error downloading {video_id}: {e}")
        import traceback
        logging.warning(f"Traceback: {traceback.format_exc()}")
        return False


def process_video_entry(row: dict, temp_dir: str, label_to_index: Optional[dict] = None, **kwargs) -> Optional[object]:
    """Download video temporarily, process it, and delete it.
    
    Args:
        row: CSV row with video_id, start, end, label, clip_id.
        temp_dir: Temporary directory for downloads.
        label_to_index: Dictionary mapping label strings to indices.
        **kwargs: Additional arguments for create_sequence_example.
        
    Returns:
        SequenceExample or None if processing failed.
    """
    # Extract video_id: remove the last part after underscore and .mp4 extension
    # Example: "---g-f_I2yQ_000001.mp4" -> "---g-f_I2yQ"
    video_path = row['video_path']
    video_id = video_path.rsplit('_', 1)[0]  # Split from right, keep left part
    
    start_time = int(row['start'])
    end_time = int(row['end'])
    label = row.get('label')
    clip_id = row.get('clip_id')
    
    # Create temporary file for video
    temp_video = os.path.join(temp_dir, f"{clip_id}.mp4")
    delete_temp = True
    
    try:
        # If a local videos directory is provided, prefer local files and skip download
        if FLAGS.local_videos_dir:
            # The CSV may contain filenames like '..._000001.mp4' or full paths; use basename when joining
            candidate = os.path.join(FLAGS.local_videos_dir, os.path.basename(video_path))
            if os.path.exists(candidate):
                logging.info(f"Using local video for {clip_id}: {candidate}")
                temp_video = candidate
                delete_temp = False
                
                # If local videos are already 10s clips, extract from 0-10 instead of CSV start/end
                if FLAGS.local_are_clips:
                    start_time = 0
                    end_time = 10
                    logging.info(f"Local video is pre-clipped, extracting 0-10s instead of {row['start']}-{row['end']}s")
                
                # Optionally check duration using ffprobe
                if FLAGS.check_duration and not FLAGS.local_are_clips:
                    duration = get_local_video_duration(candidate)
                    if duration is None:
                        logging.warning(f"Could not get local duration for {candidate}, attempting processing anyway")
                    elif start_time + 10 > duration:
                        logging.warning(f"Local video {candidate} is only {duration:.1f}s long, cannot extract segment at {start_time}s-{start_time+10}s")
                        return None
            else:
                # If require_local is set, skip this video instead of downloading
                if FLAGS.require_local:
                    logging.warning(f"Local video not found at {candidate} and --require_local is set, skipping {clip_id}")
                    return None
                # Otherwise fall back to download if local file not found
                logging.info(f"Local video not found at {candidate}, falling back to YouTube download for {video_id}")

        # Download video if we don't already have a local file
        if delete_temp:
            if not download_youtube_video(video_id, start_time, temp_video, check_duration=FLAGS.check_duration):
                return None

        # Process video - extract the specific segment with ffmpeg
        sequence_example = gen_module.create_sequence_example(
            video_path=temp_video,
            start_time=start_time,  # Extract from full video
            end_time=end_time,      # 10 second clips
            label=label,
            clip_id=clip_id,
            label_to_index=label_to_index,
            **kwargs
        )

        return sequence_example

    except Exception as e:
        logging.error(f"Error processing {clip_id}: {e}")
        return None

    finally:
        # Delete temporary file only if we created it in temp_dir
        try:
            if delete_temp and os.path.exists(temp_video) and os.path.dirname(temp_video) == os.path.abspath(temp_dir):
                os.remove(temp_video)
        except Exception:
            pass


def main(argv):
    del argv
    
    # Import preprocessing module AFTER flags are parsed to avoid conflicts
    global gen_module
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PreProcessing'))
    import PreProcessing.generate_audiovisual_from_file as gen_module # Keep this like this
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("yt-dlp not found. Install with: pip install yt-dlp")
        return
    
    # Read CSV
    logging.info(f"Reading CSV from {FLAGS.csv_path}")
    df = pd.read_csv(FLAGS.csv_path)
    total_examples = len(df)
    logging.info(f"Processing {total_examples} examples")
    
    # Create label-to-index mapping from unique labels in CSV
    label_to_index = None
    if 'label' in df.columns:
        unique_labels = sorted(df['label'].dropna().unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        logging.info(f"Created label mapping with {len(label_to_index)} unique labels")
        
        # Save label mapping to output directory for reference
        os.makedirs(FLAGS.output_path, exist_ok=True)
        label_mapping_path = os.path.join(FLAGS.output_path, 'label_mapping.txt')
        with open(label_mapping_path, 'w') as f:
            for label, idx in sorted(label_to_index.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{label}\n")
        logging.info(f"Saved label mapping to {label_mapping_path}")
    else:
        logging.warning("No 'label' column found in CSV")
    
    # Create output directory
    os.makedirs(FLAGS.output_path, exist_ok=True)
    
    # Create or use temporary directory
    if FLAGS.temp_dir:
        temp_dir = FLAGS.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix='vggsound_')
    
    logging.info(f"Using temporary directory: {temp_dir}")
    
    # Setup progress tracking
    if FLAGS.progress_file:
        progress_file = FLAGS.progress_file
    else:
        progress_file = os.path.join(FLAGS.output_path, '.progress.json')
    
    # Load existing progress
    processed_indices = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_indices = set(progress_data.get('processed_indices', []))
                logging.info(f"Resuming from progress file: {len(processed_indices)} already processed")
        except:
            logging.warning(f"Could not load progress file, starting fresh")
    
    def save_progress(processed_set):
        """Save progress to disk."""
        try:
            with open(progress_file, 'w') as f:
                json.dump({
                    'processed_indices': list(processed_set),
                    'successful': successful,
                    'failed': failed,
                    'last_updated': pd.Timestamp.now().isoformat()
                }, f)
        except Exception as e:
            logging.warning(f"Could not save progress: {e}")
    
    # Create shard writers
    writers = []
    shard_paths = []
    for shard_idx in range(FLAGS.num_shards):
        shard_path = os.path.join(
            FLAGS.output_path,
            f"data-{shard_idx:05d}-of-{FLAGS.num_shards:05d}.tfrecord"
        )
        shard_paths.append(shard_path)
        
        # Open writer in append mode if resuming
        writers.append(tf.io.TFRecordWriter(shard_path))
    
    # Process examples
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        # Skip if already processed
        if idx in processed_indices:
            skipped += 1
            continue
            
        shard_idx = idx % FLAGS.num_shards
        
        if idx % 10 == 0:
            logging.info(f"Processing {idx}/{total_examples} "
                        f"(success: {successful}, failed: {failed}, skipped: {skipped})")
        
        # Process video
        sequence_example = process_video_entry(
            row.to_dict(),
            temp_dir,
            label_to_index=label_to_index,
            target_fps=FLAGS.target_fps,
            decode_audio=FLAGS.decode_audio,
            audio_sample_rate=FLAGS.audio_sample_rate,
            n_mels=FLAGS.n_mels,
            win_length_ms=FLAGS.win_length_ms,
            hop_length_ms=FLAGS.hop_length_ms
        )
        
        if sequence_example is not None:
            writers[shard_idx].write(sequence_example.SerializeToString())
            successful += 1
        else:
            failed += 1
        
        # Mark as processed
        processed_indices.add(idx)
        
        # Save progress periodically
        if idx % FLAGS.save_progress_every == 0:
            save_progress(processed_indices)
        
        # Clean temp directory periodically
        if idx % FLAGS.batch_size == 0 and os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
    
    # Close writers
    for writer in writers:
        if writer is not None:
            writer.close()
    
    # Save final progress
    save_progress(processed_indices)
    
    # Cleanup temp directory
    if not FLAGS.temp_dir:  # Only cleanup if we created it
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
    
    logging.info("\n=== Processing Complete ===")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Skipped (existing shards): {skipped}")
    logging.info(f"Output: {FLAGS.output_path}")
    logging.info(f"\nStorage used: Only TFRecords (~{successful * 5}MB estimated)")


if __name__ == '__main__':
    app.run(main)
