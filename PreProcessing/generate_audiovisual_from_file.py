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
from typing import Dict, List, Optional, Union
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
    flags.DEFINE_float('clip_duration', 10.0, 'Duration of audio clip to extract in seconds (e.g., 8.0 for MBT AudioSet, 10.0 for VGGSound).')
    flags.DEFINE_float('rgb_duration', None, 'Duration of RGB clip to extract in seconds. If not set, uses clip_duration. For MBT AudioSet: use 3.0 for RGB, 8.0 for audio.')
    
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
    
    From the MBT paper: "we extract log mel spectrograms with a frequency dimension 
    of 128 computed using a 25ms Hamming window with hop length 10ms"
    
    However, we also apply Lingvo-specific preprocessing (preemphasis, frequency range)
    since MBT was trained using Lingvo's audio frontend.
    
    Args:
        audio: Audio waveform (mono).
        sample_rate: Audio sample rate.
        n_mels: Number of mel frequency bins.
        win_length_ms: Window length in milliseconds.
        hop_length_ms: Hop length in milliseconds.
        
    Returns:
        Linear-scale mel spectrogram of shape (time_frames, n_mels).
    """
    if librosa is None:
        raise ImportError("librosa is required for spectrogram extraction. "
                         "Install with: pip install librosa")
    
    # CRITICAL PREPROCESSING STEPS (based on Lingvo + MBT paper)
    # Reference: https://github.com/tensorflow/lingvo/blob/master/lingvo/tasks/asr/frontend.py
    
    # 1. Apply preemphasis (Lingvo default: 0.97)
    # This high-pass filter emphasizes high frequencies
    # Formula: signal[n] - preemph * signal[n-1]
    preemph = 0.97
    audio_preemph = np.append(audio[0], audio[1:] - preemph * audio[:-1])
    
    # 2. Convert ms to samples
    hop_length = int(hop_length_ms * sample_rate / 1000)  # 10ms = 160 samples
    win_length = int(win_length_ms * sample_rate / 1000)  # 25ms = 400 samples
    
    # 3. Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_preemph,
        sr=sample_rate,
        n_fft=512,  # Standard FFT size (Lingvo uses 512 or 1024 if fft_overdrive=True)
        hop_length=hop_length,
        win_length=win_length,
        window='hamming',  # PAPER EXPLICITLY STATES HAMMING WINDOW
        n_mels=n_mels,
        fmin=125.0,  # Lingvo default: 125 Hz (not 0!) - ignores very low frequencies
        fmax=7600.0,  # Lingvo default: 7600 Hz (not 8000!) - typical speech range
        power=2.0  # Energy (squared magnitude)
    )
    
    # Convert to LOG scale as per MBT paper: "log mel spectrograms"
    # Use librosa's power_to_db which converts power spectrogram to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Transpose to (time, frequency)
    mel_spec_db = mel_spec_db.T
    
    return mel_spec_db


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
                           label: Optional[Union[str, np.ndarray]] = None, 
                           caption: Optional[str] = None,
                           clip_id: Optional[str] = None,
                           target_fps: int = 25,
                           decode_audio: bool = True,
                           audio_sample_rate: int = 16000,
                           n_mels: int = 128,
                           win_length_ms: float = 25.0,
                           hop_length_ms: float = 10.0,
                           min_resize: int = 256,
                           clip_duration: float = 10.0,
                           rgb_duration: Optional[float] = None,
                           label_to_index: Optional[Dict[str, int]] = None) -> tf.train.SequenceExample:
    """Create a SequenceExample for one video clip.
    
    Args:
        video_path: Path to video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds (ignored if clip_duration is set).
        label: Optional label - either string (single-label) or numpy array (multi-hot AudioSet).
        caption: Optional caption string.
        clip_id: Optional clip identifier.
        target_fps: Target frames per second.
        decode_audio: Whether to extract audio spectrogram.
        audio_sample_rate: Audio sample rate.
        n_mels: Number of mel bins.
        win_length_ms: Window length in ms.
        hop_length_ms: Hop length in ms.
        min_resize: Resize frames so shortest side equals this (default 256).
        clip_duration: Duration of audio clip to extract (overrides end_time if set).
        rgb_duration: Duration of RGB clip to extract. If None, uses clip_duration.
        label_to_index: Dictionary mapping label strings to indices (only for string labels).
        
    Returns:
        A tf.train.SequenceExample.
    """
    # Use separate durations for RGB and audio if specified
    # This is needed because MBT can use different temporal windows for different modalities
    audio_end_time = start_time + clip_duration
    rgb_end_time = start_time + (rgb_duration if rgb_duration is not None else clip_duration)
    
    # Extract RGB frames with aspect-ratio-preserving resize
    frames = extract_frames_ffmpeg(video_path, start_time, rgb_end_time, target_fps, min_resize)
    
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    
    # Create context features
    context_dict = {}
    
    if clip_id:
        context_dict['clip/media_id'] = create_bytes_feature(clip_id.encode('utf-8'))
    
    if label is not None:
        # Handle multi-hot label (numpy array) vs string label
        if isinstance(label, np.ndarray):
            # Multi-hot label (AudioSet): store as float array
            context_dict['clip/label/multi_hot'] = create_float_feature(label.astype(np.float32).tolist())
            # Store number of active labels for reference
            num_active = int(label.sum())
            context_dict['clip/label/num_active'] = create_int64_feature(num_active)
            logging.info(f"Storing multi-hot label with {num_active} active classes")
        else:
            # String label (single-label datasets): store string and index
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
    context_dict['clip/end/timestamp'] = create_int64_feature(int(audio_end_time * 1000000))
    
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
        audio = extract_audio_ffmpeg(video_path, start_time, audio_end_time, audio_sample_rate)
        
        if audio is not None and len(audio) > 0:
            try:
                # Compute log mel spectrogram
                log_mel_spec = compute_log_mel_spectrogram(
                    audio, audio_sample_rate, n_mels, win_length_ms, hop_length_ms)
                
                # Store spectrogram as chunks of (100, n_mels) instead of (1, n_mels)
                # This matches the MBT config expectation of spec_shape=(100, 128)
                chunk_size = 100
                total_frames = log_mel_spec.shape[0]
                
                # Calculate expected frames based on clip_duration
                # For 16kHz audio with 10ms hop: expected = clip_duration * 100
                expected_frames = int(clip_duration * 100)
                
                # Truncate or pad to exact expected length
                if total_frames > expected_frames:
                    log_mel_spec = log_mel_spec[:expected_frames, :]
                    total_frames = expected_frames
                elif total_frames < expected_frames:
                    padding = np.zeros((expected_frames - total_frames, n_mels), dtype=log_mel_spec.dtype)
                    log_mel_spec = np.vstack([log_mel_spec, padding])
                    total_frames = expected_frames
                
                # Reshape into chunks of 100 frames (should divide evenly now)
                num_chunks = total_frames // chunk_size
                log_mel_spec = log_mel_spec[:num_chunks * chunk_size, :]  # Ensure exact division
                
                # Reshape to (num_chunks, chunk_size, n_mels)
                log_mel_spec_chunked = log_mel_spec.reshape(num_chunks, chunk_size, n_mels)
                
                # Store each chunk as a flat feature
                spec_list = []
                for chunk in log_mel_spec_chunked:
                    # Flatten the chunk: (100, 128) -> (12800,)
                    spec_list.append(create_float_feature(chunk.flatten().tolist()))
                
                feature_lists['WAVEFORM/feature/floats'] = tf.train.FeatureList(feature=spec_list)
                context_dict['WAVEFORM/num_mel_bins'] = create_int64_feature(n_mels)
                context_dict['WAVEFORM/sample_rate'] = create_int64_feature(audio_sample_rate)
                context_dict['WAVEFORM/chunk_size'] = create_int64_feature(chunk_size)
                
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
            hop_length_ms=FLAGS.hop_length_ms,
            clip_duration=FLAGS.clip_duration,
            rgb_duration=FLAGS.rgb_duration
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
