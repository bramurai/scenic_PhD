"""Generate audiovisual TFRecords for MBT training.

This script extends the DMVR preprocessing pipeline to extract:
1. RGB frames at 25fps (as per DMVR standard)
2. Log mel spectrograms with MBT paper specifications:
   - 16kHz sampling rate, mono channel
   - 128 mel bins
   - 25ms Hamming window, 10ms hop length
   - Results in 128 × 100t for t seconds

Usage:
  python generate_audiovisual_from_file.py \
    --csv_path=/path/to/input.csv \
    --output_path=/path/to/output \
    --decode_audio=True
"""

import os
import csv
import math
from typing import Dict, List, Optional
from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import ffmpeg

try:
    import librosa
except ImportError:
    logging.warning("librosa not installed. Install with: pip install librosa")
    librosa = None

# Only define FLAGS if this is run as a script, not when imported as a module
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('csv_path', None, 'Path to input CSV file.')
    flags.DEFINE_string('output_path', None, 'Path to output TFRecord files.')
    flags.DEFINE_string('video_root_path', '', 'Root path for video files (optional).')
    flags.DEFINE_integer('num_shards', 10, 'Number of output shards.')
    flags.DEFINE_bool('decode_audio', True, 'Whether to decode and store audio spectrograms.')
    flags.DEFINE_bool('shuffle_csv', False, 'Whether to shuffle input CSV.')
    flags.DEFINE_integer('target_fps', 25, 'Target frames per second for video.')
    flags.DEFINE_integer('audio_sample_rate', 16000, 'Audio sample rate in Hz.')
    flags.DEFINE_integer('n_mels', 128, 'Number of mel frequency bins.')
    flags.DEFINE_float('win_length_ms', 25.0, 'Window length in milliseconds.')
    flags.DEFINE_float('hop_length_ms', 10.0, 'Hop length in milliseconds.')
    
    flags.mark_flag_as_required('csv_path')
    flags.mark_flag_as_required('output_path')


def extract_frames_ffmpeg(video_path: str, start_time: float, end_time: float,
                          target_fps: int = 25, min_resize: int = 256) -> List[np.ndarray]:
    """Extract RGB frames from video using ffmpeg.
    
    Args:
        video_path: Path to video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        target_fps: Target frames per second.
        min_resize: Resize so shortest side equals this (default 256, preserves aspect ratio).
        
    Returns:
        List of RGB frames as numpy arrays.
    """
    try:
        # Extract frames with resizing (shortest side = min_resize, preserving aspect ratio)
        # ffmpeg scale=-1:256 means: height=256, width=auto (preserve aspect)
        # ffmpeg scale=256:-1 means: width=256, height=auto (preserve aspect)
        # We use scale=-2:min_resize for portrait, scale=min_resize:-2 for landscape
        # -2 ensures dimensions are divisible by 2 (required for many codecs)
        
        duration = end_time - start_time
        
        # First get video dimensions to determine orientation
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Scale so shorter side = min_resize
        if width < height:
            # Portrait/vertical: width is shorter
            new_width = min_resize
            new_height = int(height * min_resize / width)
            new_height = new_height - (new_height % 2)  # Make even
            scale_w, scale_h = new_width, new_height
        else:
            # Landscape/horizontal: height is shorter
            new_height = min_resize
            new_width = int(width * min_resize / height)
            new_width = new_width - (new_width % 2)  # Make even
            scale_w, scale_h = new_width, new_height
        
        out, _ = (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .filter('fps', fps=target_fps)
            .filter('scale', w=scale_w, h=scale_h)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # Convert to numpy array
        frames = np.frombuffer(out, np.uint8).reshape([-1, new_height, new_width, 3])
        return [frame for frame in frames]
        
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error extracting frames from {video_path}: {e.stderr.decode() if e.stderr else str(e)}")
        return []
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}")
        return []


def extract_audio_ffmpeg(video_path: str, start_time: float, end_time: float,
                         sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract audio from video using ffmpeg.
    
    Args:
        video_path: Path to video file.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        sample_rate: Target audio sample rate.
        
    Returns:
        Audio waveform as numpy array (mono, float32).
    """
    try:
        duration = end_time - start_time
        out, _ = (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output('pipe:', format='f32le', acodec='pcm_f32le', 
                   ac=1, ar=sample_rate)
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        audio = np.frombuffer(out, np.float32)
        return audio
        
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        return None


def compute_log_mel_spectrogram(audio: np.ndarray, sample_rate: int = 16000,
                                n_mels: int = 128, win_length_ms: float = 25.0,
                                hop_length_ms: float = 10.0) -> np.ndarray:
    """Compute log mel spectrogram following MBT paper specifications.
    
    Args:
        audio: Audio waveform (mono).
        sample_rate: Audio sample rate.
        n_mels: Number of mel frequency bins.
        win_length_ms: Window length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        
    Returns:
        Log mel spectrogram of shape (time_frames, n_mels).
    """
    if librosa is None:
        raise ImportError("librosa is required for spectrogram extraction. "
                         "Install with: pip install librosa")
    
    # Convert ms to samples
    win_length = int(win_length_ms * sample_rate / 1000)
    hop_length = int(hop_length_ms * sample_rate / 1000)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=512,
        hop_length=hop_length,
        win_length=win_length,
        window='hamming',
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate // 2
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to (time, frequency)
    log_mel_spec = log_mel_spec.T
    
    return log_mel_spec


def create_int64_feature(value):
    """Create an int64 feature."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_float_feature(value):
    """Create a float feature."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_bytes_feature(value):
    """Create a bytes feature."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_sequence_example(video_path: str, start_time: float, end_time: float,
                           label: Optional[str] = None, 
                           caption: Optional[str] = None,
                           clip_id: Optional[str] = None,
                           target_fps: int = 25,
                           decode_audio: bool = True,
                           audio_sample_rate: int = 16000,
                           n_mels: int = 128,
                           win_length_ms: float = 25.0,
                           hop_length_ms: float = 10.0,
                           min_resize: int = 256,
                           label_to_index: Optional[Dict[str, int]] = None) -> tf.train.SequenceExample:
    """Create a SequenceExample for one video clip.
    
    Args:
        video_path: Path to video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        label: Optional label string.
        caption: Optional caption string.
        clip_id: Optional clip identifier.
        target_fps: Target frames per second.
        decode_audio: Whether to extract audio spectrogram.
        audio_sample_rate: Audio sample rate.
        n_mels: Number of mel bins.
        win_length_ms: Window length in ms.
        hop_length_ms: Hop length in ms.
        min_resize: Resize frames so shortest side equals this (default 256).
        label_to_index: Dictionary mapping label strings to indices (required if label is provided).
        
    Returns:
        A tf.train.SequenceExample.
    """
    # Extract RGB frames with aspect-ratio-preserving resize
    frames = extract_frames_ffmpeg(video_path, start_time, end_time, target_fps, min_resize)
    
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    
    # Create context features
    context_dict = {}
    
    if clip_id:
        context_dict['clip/media_id'] = create_bytes_feature(clip_id.encode('utf-8'))
    
    if label is not None:
        context_dict['clip/label/string'] = create_bytes_feature(label.encode('utf-8'))
        
        # Use label_to_index mapping if provided, otherwise use placeholder
        if label_to_index is not None:
            if label in label_to_index:
                label_idx = label_to_index[label]
            else:
                logging.warning(f"Label '{label}' not found in label_to_index mapping, using -1")
                label_idx = -1
        else:
            logging.warning("No label_to_index mapping provided, all labels will be mapped to 0")
            label_idx = 0
        
        context_dict['clip/label/index'] = create_int64_feature(label_idx)
    
    if caption is not None:
        context_dict['clip/caption/string'] = create_bytes_feature(caption.encode('utf-8'))
    
    context_dict['clip/start/timestamp'] = create_int64_feature(int(start_time * 1000000))
    context_dict['clip/end/timestamp'] = create_int64_feature(int(end_time * 1000000))
    
    # Create feature lists
    feature_lists = {}
    
    # Add RGB frames
    frame_list = []
    for frame in frames:
        # Encode frame as JPEG
        frame_encoded = tf.image.encode_jpeg(frame).numpy()
        frame_list.append(create_bytes_feature(frame_encoded))
    feature_lists['image/encoded'] = tf.train.FeatureList(feature=frame_list)
    feature_lists['image/height'] = tf.train.FeatureList(
        feature=[create_int64_feature(frames[0].shape[0])] * len(frames))
    feature_lists['image/width'] = tf.train.FeatureList(
        feature=[create_int64_feature(frames[0].shape[1])] * len(frames))
    
    # Extract and add spectrogram if requested
    if decode_audio:
        audio = extract_audio_ffmpeg(video_path, start_time, end_time, audio_sample_rate)
        
        if audio is not None and len(audio) > 0:
            try:
                # Compute log mel spectrogram
                log_mel_spec = compute_log_mel_spectrogram(
                    audio, audio_sample_rate, n_mels, win_length_ms, hop_length_ms)
                
                # Store spectrogram as a sequence of float features
                # Shape: (time_frames, n_mels) -> flatten each time frame
                spec_list = []
                for spec_frame in log_mel_spec:
                    spec_list.append(create_float_feature(spec_frame.tolist()))
                
                feature_lists['WAVEFORM/feature/floats'] = tf.train.FeatureList(feature=spec_list)
                context_dict['WAVEFORM/num_mel_bins'] = create_int64_feature(n_mels)
                context_dict['WAVEFORM/sample_rate'] = create_int64_feature(audio_sample_rate)
                
            except Exception as e:
                logging.warning(f"Error computing spectrogram for {video_path}: {e}")
    
    # Create SequenceExample
    context = tf.train.Features(feature=context_dict)
    feature_list_dict = tf.train.FeatureLists(feature_list=feature_lists)
    sequence_example = tf.train.SequenceExample(
        context=context,
        feature_lists=feature_list_dict)
    
    return sequence_example


def process_csv_row(row: Dict, video_root_path: str, label_to_index: Optional[Dict[str, int]] = None, **kwargs) -> Optional[tf.train.SequenceExample]:
    """Process one row from the CSV file.
    
    Args:
        row: Dictionary with CSV row data.
        video_root_path: Root path for video files.
        label_to_index: Dictionary mapping label strings to indices.
        **kwargs: Additional arguments for create_sequence_example.
        
    Returns:
        A SequenceExample or None if processing failed.
    """
    video_path = row['video_path']
    if video_root_path:
        video_path = os.path.join(video_root_path, video_path)
    
    start_time = float(row['start'])
    end_time = float(row['end'])
    label = row.get('label')
    caption = row.get('caption')
    clip_id = row.get('clip_id', f"{os.path.basename(video_path)}_{start_time}_{end_time}")
    
    try:
        return create_sequence_example(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            label=label,
            caption=caption,
            clip_id=clip_id,
            label_to_index=label_to_index,
            **kwargs
        )
    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return None


def main(argv):
    del argv  # Unused.
    
    # Read CSV
    logging.info(f"Reading CSV from {FLAGS.csv_path}")
    df = pd.read_csv(FLAGS.csv_path)
    
    if FLAGS.shuffle_csv:
        df = df.sample(frac=1).reset_index(drop=True)
    
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
    
    # Create shard writers
    writers = []
    for shard_idx in range(FLAGS.num_shards):
        shard_path = os.path.join(
            FLAGS.output_path,
            f"data-{shard_idx:05d}-of-{FLAGS.num_shards:05d}.tfrecord"
        )
        writers.append(tf.io.TFRecordWriter(shard_path))
    
    # Process examples
    successful = 0
    failed = 0
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logging.info(f"Processing example {idx}/{total_examples}")
        
        sequence_example = process_csv_row(
            row.to_dict(),
            FLAGS.video_root_path,
            label_to_index=label_to_index,
            target_fps=FLAGS.target_fps,
            decode_audio=FLAGS.decode_audio,
            audio_sample_rate=FLAGS.audio_sample_rate,
            n_mels=FLAGS.n_mels,
            win_length_ms=FLAGS.win_length_ms,
            hop_length_ms=FLAGS.hop_length_ms
        )
        
        if sequence_example is not None:
            # Write to shard (round-robin)
            shard_idx = idx % FLAGS.num_shards
            writers[shard_idx].write(sequence_example.SerializeToString())
            successful += 1
        else:
            failed += 1
    
    # Close writers
    for writer in writers:
        writer.close()
    
    logging.info(f"Processing complete!")
    logging.info(f"  Successful: {successful}")
    logging.info(f"  Failed: {failed}")
    logging.info(f"  Output: {FLAGS.output_path}")


if __name__ == '__main__':
    app.run(main)
