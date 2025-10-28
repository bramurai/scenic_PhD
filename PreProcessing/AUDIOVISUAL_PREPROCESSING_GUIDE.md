# Audiovisual TFRecord Preprocessing Guide for MBT

This guide explains how to preprocess your video dataset into TFRecords with both RGB frames and audio spectrograms for MBT training.

## Requirements

Install the required packages:

```bash
pip install librosa ffmpeg-python pandas tensorflow absl-py
```

Also install ffmpeg binaries (if not already installed):
- Download from: https://johnvansickle.com/ffmpeg/
- Add to PATH

## Step 1: Prepare Your CSV File

Create a CSV file with the following columns:

| Column      | Description                          | Required |
|-------------|--------------------------------------|----------|
| video_path  | Path to video file                   | Yes      |
| start       | Start time in seconds                | Yes      |
| end         | End time in seconds                  | Yes      |
| label       | Class label (string)                 | No       |
| caption     | Text caption                         | No       |
| clip_id     | Unique clip identifier               | No       |

### Example CSV for VGGSound:

```csv
video_path,start,end,label
videos/--PJHxphWEs_000001.mp4,0,10,dog_barking
videos/--ZhevVpy1s_000321.mp4,0,10,violin
videos/-0RWZT-miFs_000030.mp4,0,10,ambulance_siren
```

### Example CSV with Custom Clips:

```csv
video_path,start,end,label,caption
/path/to/video1.mp4,5.0,15.0,action_1,person running
/path/to/video2.mp4,2.5,12.5,action_2,person jumping
```

## Step 2: Run Preprocessing

### Basic Usage:

```bash
python generate_audiovisual_from_file.py \
  --csv_path=/path/to/your/dataset.csv \
  --output_path=/path/to/output/tfrecords \
  --decode_audio=True \
  --num_shards=100
```

### With Video Root Path:

If your CSV contains relative paths, specify a root directory:

```bash
python generate_audiovisual_from_file.py \
  --csv_path=vggsound_train.csv \
  --video_root_path=/data/vggsound/videos \
  --output_path=/data/vggsound/tfrecords/train \
  --decode_audio=True \
  --num_shards=100
```

### All Available Flags:

| Flag                  | Default | Description                                    |
|-----------------------|---------|------------------------------------------------|
| `--csv_path`          | *       | Path to input CSV (required)                   |
| `--output_path`       | *       | Output directory for TFRecords (required)      |
| `--video_root_path`   | ""      | Root path prepended to video_path in CSV       |
| `--num_shards`        | 10      | Number of TFRecord shards to create            |
| `--decode_audio`      | True    | Extract audio spectrograms                     |
| `--shuffle_csv`       | False   | Shuffle input CSV before processing            |
| `--target_fps`        | 25      | Target FPS for frame extraction                |
| `--audio_sample_rate` | 16000   | Audio sample rate (Hz)                         |
| `--n_mels`            | 128     | Number of mel frequency bins                   |
| `--win_length_ms`     | 25.0    | Spectrogram window length (ms)                 |
| `--hop_length_ms`     | 10.0    | Spectrogram hop length (ms)                    |

## Step 3: Verify Output

The script will create sharded TFRecord files:

```
/path/to/output/tfrecords/
├── data-00000-of-00100.tfrecord
├── data-00001-of-00100.tfrecord
├── ...
└── data-00099-of-00100.tfrecord
```

### Verify a TFRecord:

```python
import tensorflow as tf

# Load one TFRecord
dataset = tf.data.TFRecordDataset(['data-00000-of-00100.tfrecord'])

for raw_record in dataset.take(1):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    # Check RGB frames
    num_frames = len(example.feature_lists.feature_list['image/encoded'].feature)
    print(f"Number of RGB frames: {num_frames}")
    
    # Check spectrogram
    if 'WAVEFORM/feature/floats' in example.feature_lists.feature_list:
        num_spec_frames = len(example.feature_lists.feature_list['WAVEFORM/feature/floats'].feature)
        print(f"Number of spectrogram frames: {num_spec_frames}")
        
        n_mels = example.context.feature['WAVEFORM/num_mel_bins'].int64_list.value[0]
        print(f"Mel bins: {n_mels}")
```

## Step 4: Configure MBT Dataset Loader

Update your MBT config to point to the TFRecords:

```python
# In your config file (e.g., balanced_audioset_base.py)

def get_config():
    config = ml_collections.ConfigDict()
    
    # ... other config ...
    
    # Dataset paths
    config.dataset_configs.base_dir = '/path/to/output/tfrecords'
    config.dataset_configs.tables = {
        'train': 'train/data@100',  # If you have 100 shards
        'validation': 'val/data@10',
        'test': 'test/data@10',
    }
    
    # Make sure these match your preprocessing settings
    config.dataset_configs.num_frames = 8  # Or 32 for Epic-Kitchens
    config.dataset_configs.stride = 32  # Depends on clip length
    config.dataset_configs.num_spec_frames = 10  # For 10 second clips
    
    return config
```

## Expected Data Format

After preprocessing, each TFRecord entry contains:

**Context Features:**
- `clip/media_id`: Unique clip identifier
- `clip/label/string`: Label string
- `clip/start/timestamp`: Start timestamp (microseconds)
- `clip/end/timestamp`: End timestamp (microseconds)
- `WAVEFORM/num_mel_bins`: 128
- `WAVEFORM/sample_rate`: 16000

**Feature Lists:**
- `image/encoded`: JPEG-encoded RGB frames (list)
- `image/height`: Frame height (list, repeated)
- `image/width`: Frame width (list, repeated)
- `WAVEFORM/feature/floats`: Spectrogram frames (list of float arrays)

**Spectrogram Shape:**
- Each frame: 128 mel bins (floats)
- Total frames: ~100 per second (with 10ms hop)
- For 10-second clip: ~1000 spectrogram frames

## Troubleshooting

### Error: "No frames extracted"
- Check that video file exists and is readable
- Check start/end times are valid for the video
- Verify ffmpeg is installed and in PATH

### Error: "librosa not installed"
- Install with: `pip install librosa`

### Error: "ffmpeg not found"
- Download ffmpeg binaries from https://johnvansickle.com/ffmpeg/
- Add to system PATH
- Install python wrapper: `pip install ffmpeg-python`

### Slow processing
- Use `--num_shards` to parallelize (process different shards separately)
- Reduce video resolution before preprocessing
- Use SSD for faster I/O

## Performance Tips

1. **Parallel Processing**: Process different shards in parallel:
   ```bash
   # Split CSV into chunks and run multiple processes
   python generate_audiovisual_from_file.py --csv_path=chunk_0.csv --output_path=out --num_shards=10 &
   python generate_audiovisual_from_file.py --csv_path=chunk_1.csv --output_path=out --num_shards=10 &
   ```

2. **Storage**: TFRecords with audio are large (~5-10MB per 10-second clip)
   - Budget ~1GB per 100-200 clips
   - For VGGSound (200k clips): expect ~1-2TB

3. **Memory**: Processing is streaming-based, memory usage stays low
   - Typical: <2GB RAM per process

## Next Steps

After preprocessing:
1. Update your MBT config with TFRecord paths
2. Verify the dataset loads correctly with a small training run
3. Train your MBT model!

See [`GPU_TRAINING_GUIDE.md`](scenic/projects/mbt/GPU_TRAINING_GUIDE.md ) for training instructions.
