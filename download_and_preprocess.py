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
flags.DEFINE_integer('save_progress_every', 1, 'Save progress every N videos.')
flags.DEFINE_bool('check_duration', False, 'Verify video duration before downloading segment.')
flags.DEFINE_string('cookies_from_browser', None, 'Browser to extract cookies from (chrome, firefox, edge, etc).')
flags.DEFINE_string('cookies_file', None, 'Path to cookies.txt file for yt-dlp authentication.')
flags.DEFINE_string('local_videos_dir', None, 'If set, treat video paths as relative to this directory and use local mp4 files instead of downloading from YouTube.')
flags.DEFINE_bool('require_local', False, 'If True and local_videos_dir is set, skip videos not found locally instead of falling back to YouTube download.')
flags.DEFINE_bool('local_are_clips', False, 'If True, local videos are already 10s clips (extract 0-10s instead of using CSV start/end times).')
flags.DEFINE_string('audioset_labels_csv', None, 'Path to audioset_labels.csv for mapping MIDs to indices (enables multi-hot label encoding for AudioSet).')
flags.DEFINE_bool('download_full_segment', True, 'If True, download the entire CSV segment (start to end). If False, download full video.')

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


def load_audioset_labels(audioset_labels_csv: str) -> dict:
    """Load AudioSet label mapping from CSV.
    
    Args:
        audioset_labels_csv: Path to audioset_labels.csv with columns: index, mid, display_name
        
    Returns:
        Dictionary mapping MID to index, e.g., {'/m/09x0r': 0, '/m/05zppz': 1, ...}
    """
    import pandas as pd
    
    logging.info(f'Loading AudioSet labels from {audioset_labels_csv}')
    df = pd.read_csv(audioset_labels_csv)
    
    # Create MID to index mapping
    mid_to_index = {}
    for _, row in df.iterrows():
        mid = row['mid']
        index = int(row['index'])
        mid_to_index[mid] = index
    
    logging.info(f'Loaded {len(mid_to_index)} AudioSet labels')
    return mid_to_index


def parse_audioset_labels(label_string: str, mid_to_index: dict, num_classes: int = 527) -> np.ndarray:
    """Parse AudioSet label string into multi-hot encoding.
    
    Args:
        label_string: Comma-separated MIDs like "/m/068hy,/m/07q6cd_,/m/0bt9lr"
        mid_to_index: Dictionary mapping MID to index
        num_classes: Total number of AudioSet classes (default 527)
        
    Returns:
        Multi-hot label array of shape (num_classes,) with 1s for present classes
    """
    import numpy as np
    
    # Create zero array
    multi_hot = np.zeros(num_classes, dtype=np.float32)
    
    # Parse comma-separated MIDs
    mids = [mid.strip() for mid in label_string.split(',')]
    
    # Set 1s for present labels
    for mid in mids:
        if mid in mid_to_index:
            index = mid_to_index[mid]
            multi_hot[index] = 1.0
        else:
            logging.warning(f'Unknown MID: {mid}')
    
    return multi_hot


def download_youtube_video(video_id: str, start_time: int, end_time: int, output_path: str, 
                          check_duration: bool = True, download_segment_only: bool = True) -> bool:
    """Download a video or segment from YouTube using yt-dlp.
    
    Args:
        video_id: YouTube video ID.
        start_time: Start time in seconds from CSV.
        end_time: End time in seconds from CSV.
        output_path: Output file path.
        check_duration: Whether to verify video duration before downloading.
        download_segment_only: If True, download only the CSV segment. If False, download full video.
        
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
            elif end_time > duration:
                logging.warning(f"Video {video_id} is only {duration:.1f}s long, "
                              f"cannot extract segment at {start_time}s-{end_time}s")
                return False
        
        # Build yt-dlp command
        cmd = [
            'yt-dlp',
            '--quiet',
            '--no-warnings',
            '--format', 'worst[ext=mp4]/worst',  # Lowest quality, prefer mp4
            '--output', output_path,
            '--no-playlist',
            '--concurrent-fragments', '4',  # Download fragments in parallel
            '--throttled-rate', '100K',  # Skip if speed drops below 100KB/s
            '--sleep-requests', '1',  # Sleep 1 second between requests
        ]
        
        # Add segment download if requested (much faster and less storage)
        if download_segment_only:
            segment_duration = end_time - start_time
            cmd.extend([
                '--download-sections', f'*{start_time}-{end_time}',
                '--force-keyframes-at-cuts',  # Ensure clean segment boundaries
            ])
            logging.info(f"Downloading {segment_duration}s segment ({start_time}-{end_time}s) from {video_id}")
        else:
            logging.info(f"Downloading full video {video_id}")
        
        # Add cookie authentication if provided
        if FLAGS.cookies_from_browser:
            cmd.extend(['--cookies-from-browser', FLAGS.cookies_from_browser])
        elif FLAGS.cookies_file:
            if not os.path.exists(FLAGS.cookies_file):
                logging.error(f"Cookies file not found: {FLAGS.cookies_file}")
                return False
            cmd.extend(['--cookies', FLAGS.cookies_file])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            # Verify downloaded file has reasonable size (not empty/corrupted)
            file_size = os.path.getsize(output_path)
            if file_size < 1000:  # Less than 1KB is suspicious
                logging.warning(f"Downloaded file for {video_id} is too small ({file_size} bytes)")
                return False
            logging.info(f"Successfully downloaded {video_id} ({file_size / 1024**2:.1f} MB)")
            return True
        else:
            # Log detailed error for debugging
            error_msg = result.stderr if result.stderr else result.stdout if result.stdout else "Unknown error"
            logging.warning(f"Failed to download {video_id} at {start_time}s")
            logging.warning(f"yt-dlp error: {error_msg}")
            logging.warning(f"Return code: {result.returncode}")
            
            # Check for age-restricted videos (not critical, just skip them)
            if "confirm your age" in error_msg.lower() or "age" in error_msg.lower() and "inappropriate" in error_msg.lower():
                logging.warning(f"Video {video_id} is age-restricted, skipping")
                return False
            
            # Check for ACTUAL bot detection error - this requires user intervention
            # Only raise RuntimeError if it's asking to sign in AND we don't already have cookies configured
            if "Sign in to confirm you're not a bot" in error_msg:
                if not FLAGS.cookies_from_browser and not FLAGS.cookies_file:
                    raise RuntimeError(
                        "YouTube bot detection triggered! YouTube is asking for authentication.\n"
                        "To fix this:\n"
                        "1. Export your YouTube cookies using: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies\n"
                        "2. Restart the script with --cookies_file=/path/to/cookies.txt or --cookies_from_browser=chrome\n"
                        "Progress has been saved and will resume from where it left off."
                    )
                else:
                    # We have cookies but still got bot detection - this is serious
                    logging.warning(f"Bot detection despite having cookies for {video_id}, your cookies may have expired")
                    return False
            
            return False
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout downloading {video_id} after 300 seconds")
        return False
    except RuntimeError:
        # Re-raise RuntimeError for bot detection to propagate to main()
        raise
    except Exception as e:
        logging.warning(f"Error downloading {video_id}: {e}")
        import traceback
        logging.warning(f"Traceback: {traceback.format_exc()}")
        return False


def process_video_entry(row: dict, temp_dir: str, label_to_index: Optional[dict] = None, 
                        mid_to_index: Optional[dict] = None, **kwargs) -> Optional[object]:
    """Download video and process it (videos are always kept for reuse).
    
    Args:
        row: CSV row with video_id, start, end, label, clip_id.
        temp_dir: Directory for downloaded videos.
        label_to_index: Dictionary mapping label strings to indices (for single-label datasets).
        mid_to_index: Dictionary mapping AudioSet MIDs to indices (for AudioSet multilabel).
        **kwargs: Additional arguments for create_sequence_example.
        
    Returns:
        SequenceExample or None if processing failed.
    """
    # Extract video_id: remove the last part after underscore and .mp4 extension
    # Example: "---g-f_I2yQ_000001.mp4" -> "---g-f_I2yQ"
    video_path = row['video_path']
    video_id = video_path.rsplit('_', 1)[0]  # Split from right, keep left part
    
    csv_start_time = int(row['start'])
    csv_end_time = int(row['end'])
    label = row.get('label')
    clip_id = row.get('clip_id')
    
    # Create file path for video (always kept, never deleted)
    temp_video = os.path.join(temp_dir, f"{clip_id}.mp4")
    need_to_download = True  # Assume we need to download unless we find a local copy
    
    # Determine extraction times based on whether we downloaded a segment or full video
    # If we download segment only, extract from beginning (0)
    # If we download full video, extract from csv_start_time
    extraction_start = 0  # Will be adjusted based on download mode
    
    # Calculate clip duration from CSV (end - start)
    clip_duration = csv_end_time - csv_start_time
    
    try:
        # Check if video already exists in temp_dir (current download directory)
        if os.path.exists(temp_video):
            logging.info(f"Found video in temp_dir for {clip_id}: {temp_video}")
            need_to_download = False
            # Downloaded segment videos should be extracted from 0
            extraction_start = 0 if FLAGS.download_full_segment else csv_start_time
        
        # Check temp_downloads/ directory (takes priority over local_videos_dir)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_downloads')):
            temp_downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_downloads')
            temp_downloads_candidate = os.path.join(temp_downloads_dir, os.path.basename(video_path))
            
            if os.path.exists(temp_downloads_candidate):
                logging.info(f"Found video in temp_downloads/ for {clip_id}: {temp_downloads_candidate}")
                temp_video = temp_downloads_candidate
                need_to_download = False
                # Assume temp_downloads contains full videos
                extraction_start = csv_start_time

        # If not in temp_dir or temp_downloads/, check local_videos_dir
        if need_to_download and FLAGS.local_videos_dir:
            # The CSV may contain filenames like '..._000001.mp4' or full paths; use basename when joining
            candidate = os.path.join(FLAGS.local_videos_dir, os.path.basename(video_path))
            if os.path.exists(candidate):
                logging.info(f"Using local video for {clip_id}: {candidate}")
                temp_video = candidate
                need_to_download = False
                
                # If local videos are already clips, extract from 0-clip_duration instead of CSV start/end
                if FLAGS.local_are_clips:
                    extraction_start = 0
                    logging.info(f"Local video is pre-clipped, extracting from 0s instead of {csv_start_time}s")
                else:
                    extraction_start = csv_start_time
            else:
                # If require_local is set, skip this video instead of downloading
                if FLAGS.require_local:
                    logging.warning(f"Local video not found at {candidate} and --require_local is set, skipping {clip_id}")
                    return None
                # Otherwise fall back to download if local file not found
                logging.info(f"Local video not found at {candidate}, falling back to YouTube download for {video_id}")

        # Download video from YouTube if we don't have a local copy
        if need_to_download:
            if not download_youtube_video(
                video_id, 
                csv_start_time, 
                csv_end_time, 
                temp_video, 
                check_duration=FLAGS.check_duration,
                download_segment_only=FLAGS.download_full_segment
            ):
                return None
            
            # If we downloaded segment only, extract from beginning (0)
            # If we downloaded full video, extract from csv_start_time
            extraction_start = 0 if FLAGS.download_full_segment else csv_start_time

        # Process label: if AudioSet multilabel mode, parse MIDs to multi-hot array
        processed_label = label
        if mid_to_index is not None and label:
            # AudioSet multi-label mode: convert MID string to multi-hot array
            processed_label = parse_audioset_labels(label, mid_to_index)
            logging.info(f"Converted label '{label}' to multi-hot array with {int(processed_label.sum())} active classes")
        
        # Process video - extract the specific segment with ffmpeg
        # extraction_start is 0 if we downloaded segment, csv_start_time if full video
        # end_time is extraction_start + clip_duration (duration from CSV)
        sequence_example = gen_module.create_sequence_example(
            video_path=temp_video,
            start_time=extraction_start,  # Start from 0 (segment) or csv_start_time (full video)
            end_time=extraction_start + clip_duration,  # End based on CSV duration
            label=processed_label,  # Either original string or multi-hot array
            clip_id=clip_id,
            label_to_index=label_to_index if mid_to_index is None else None,  # Only use label_to_index for non-AudioSet
            **kwargs
        )

        return sequence_example

    except RuntimeError:
        # Re-raise RuntimeError for bot detection to propagate to main()
        raise
    except Exception as e:
        logging.error(f"Error processing {clip_id}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


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
    
    # Read CSV with proper handling for AudioSet format
    # AudioSet CSV has comment lines (starting with #) and uses comma-space as separator
    # Format: YTID, start_seconds, end_seconds, positive_labels (no header row)
    logging.info(f"Reading CSV from {FLAGS.csv_path}")
    df = pd.read_csv(
        FLAGS.csv_path, 
        comment='#',  # Skip lines starting with #
        skipinitialspace=True,  # Handle spaces after commas
        sep=',\s+',  # Use regex to match comma followed by spaces
        engine='python',  # Required for regex separators
        header=None,  # No header in AudioSet CSV
        names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels']  # Set column names
    )
    
    # Convert to format expected by download_and_preprocess pipeline
    df = df.rename(columns={
        'YTID': 'video_path',
        'start_seconds': 'start',
        'end_seconds': 'end',
        'positive_labels': 'label'
    })
    
    # Remove quotes from positive_labels if present
    df['label'] = df['label'].str.strip('"')
    
    # Generate clip_id
    df['clip_id'] = df['video_path'] + '_' + df['start'].astype(int).astype(str)
    
    # Convert video_path to expected format (YTID -> YTID_000001.mp4)
    df['video_path'] = df['video_path'] + '_000001.mp4'
    
    total_examples = len(df)
    logging.info(f"Processing {total_examples} examples")
    
    # Handle label mapping based on mode (AudioSet multilabel vs single-label)
    label_to_index = None
    mid_to_index = None
    
    if FLAGS.audioset_labels_csv:
        # AudioSet multilabel mode: load MID to index mapping
        logging.info(f"Loading AudioSet labels from {FLAGS.audioset_labels_csv}")
        mid_to_index = load_audioset_labels(FLAGS.audioset_labels_csv)
        logging.info(f"AudioSet multilabel mode: using {len(mid_to_index)} classes")
    elif 'label' in df.columns:
        # Single-label mode: create label-to-index mapping from unique labels in CSV
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
        logging.warning("No 'label' column found in CSV and no AudioSet labels provided")
    
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
        except Exception as e:
            logging.warning(f"Could not load progress file: {e}. Starting fresh")
    
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
    
    try:
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
                mid_to_index=mid_to_index,
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
            
            # Periodically flush writers to disk to ensure data is saved
            if successful % 50 == 0:
                for writer in writers:
                    writer.flush()
            
            # Mark as processed
            processed_indices.add(idx)
            
            # Save progress periodically
            if idx % FLAGS.save_progress_every == 0:
                save_progress(processed_indices)
    
    except RuntimeError as e:
        # Handle bot detection or other critical errors
        if "bot detection" in str(e).lower() or "cookies" in str(e).lower():
            logging.error("\n" + "="*80)
            logging.error("CRITICAL ERROR - Authentication Required")
            logging.error("="*80)
            logging.error(str(e))
            logging.error("="*80)
            logging.error(f"\nProgress saved at: {progress_file}")
            logging.error(f"Processed so far: {successful} successful, {failed} failed")
            logging.error("\nTo resume after adding cookies, run the same command again.")
            logging.error("The script will automatically resume from where it left off.\n")
            
            # Close writers and save progress before exiting
            for writer in writers:
                if writer is not None:
                    writer.flush()
                    writer.close()
            save_progress(processed_indices)
            
            # Exit with error code
            import sys
            sys.exit(1)
        else:
            raise
    
    # Close writers
    for writer in writers:
        if writer is not None:
            writer.flush()  # Ensure all buffered data is written
            writer.close()
    
    # Save final progress
    save_progress(processed_indices)
    
    # Videos are always kept in temp_dir
    logging.info(f"Downloaded videos saved in: {temp_dir}")
    
    logging.info("\n=== Processing Complete ===")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Skipped (existing shards): {skipped}")
    logging.info(f"Output: {FLAGS.output_path}")
    logging.info(f"Videos: {temp_dir}")


if __name__ == '__main__':
    app.run(main)
