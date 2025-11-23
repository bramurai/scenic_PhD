# AudioSet Preprocessing Guide

## Overview

This guide explains the new AudioSet multi-label preprocessing pipeline that correctly maps AudioSet MIDs (Machine IDs) to their official indices using `audioset_labels.csv`.

## Problem Background

The original preprocessing created a custom label mapping (0-76) from unique labels in the CSV, which didn't match AudioSet's official 527-class ontology. This caused label mismatches when using pretrained MBT checkpoints trained on AudioSet.

## Solution

The new pipeline:
1. Uses AudioSet's official MID-to-index mapping from `audioset_labels.csv`
2. Supports multi-label encoding (each sample can have multiple classes)
3. Creates multi-hot vectors (527-dimensional binary arrays)
4. Stores labels as float arrays in TFRecords

## Files Modified

### 1. `create_audioset_ready.py`
**Purpose**: Transform `audioset_eval.csv` into preprocessing-compatible format

**Transformations**:
- `YTID` → `video_path`: Appends "_000001.mp4"
- `start_seconds` → `start`: Unchanged
- `end_seconds` → `end`: Recalculated as `start + 8.0` seconds
- `positive_labels` → `label`: Preserved as comma-separated MIDs
- Generates `clip_id` column

**Usage**:
```bash
python create_audioset_ready.py
```

**Output**: `Video_csvs/audioset_ready.csv` with 20,371 examples

### 2. `download_and_preprocess.py`
**Purpose**: Download videos and create TFRecords with proper AudioSet labels

**New Flags**:
- `--audioset_labels_csv`: Path to audioset_labels.csv (enables AudioSet mode)
- `--audioset_multilabel`: Boolean flag (currently informational)

**New Functions**:

```python
def load_audioset_labels(audioset_labels_csv: str) -> dict:
    """Load AudioSet MID to index mapping.
    
    Returns:
        Dictionary {mid: index}, e.g., {'/m/09x0r': 0, '/m/015p6': 111, ...}
    """
```

```python
def parse_audioset_labels(label_string: str, mid_to_index: dict, num_classes: int = 527):
    """Convert comma-separated MIDs to multi-hot array.
    
    Args:
        label_string: e.g., "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"
        mid_to_index: Dictionary mapping MIDs to indices
        num_classes: Total number of AudioSet classes (default 527)
    
    Returns:
        numpy array of shape (527,) with 1.0 for present classes
    """
```

**Modified Functions**:
- `main()`: Loads AudioSet labels if `--audioset_labels_csv` provided
- `process_video_entry()`: Calls `parse_audioset_labels()` for multi-hot encoding

### 3. `generate_audiovisual_from_file.py`
**Purpose**: Extract RGB frames and audio spectrograms, create TFRecords

**Modified Function**:

```python
def create_sequence_example(
    video_path: str, 
    start_time: float, 
    end_time: float,
    label: Optional[Union[str, np.ndarray]] = None,  # Now accepts both!
    ...
):
    """Create a SequenceExample for one video clip.
    
    Args:
        label: Either string (single-label) or numpy array (multi-hot AudioSet)
        ...
    """
```

**Label Handling**:
- If `label` is `np.ndarray`: Stores as `clip/label/multi_hot` (float feature)
- If `label` is `str`: Stores as `clip/label/string` and `clip/label/index` (original behavior)

**TFRecord Features**:
- Multi-label mode:
  - `clip/label/multi_hot`: Float array of shape (527,)
  - `clip/label/num_active`: Integer count of active classes
- Single-label mode:
  - `clip/label/string`: Label text
  - `clip/label/index`: Integer index

## Usage

### Step 1: Create AudioSet-Ready CSV

```bash
python create_audioset_ready.py
```

**Input**: `Video_csvs/audioset_eval.csv`
**Output**: `Video_csvs/audioset_ready.csv`

### Step 2: Preprocess Videos

```bash
python zLabels_preprocess/download_and_preprocess.py \
  --csv_path=Video_csvs/audioset_ready.csv \
  --output_path=Datasets/audioset_eval_fullset \
  --audioset_labels_csv=Video_csvs/audioset_labels.csv \
  --clip_duration=8.0 \
  --rgb_duration=3.0 \
  --num_shards=100 \
  --target_fps=25 \
  --decode_audio=True \
  --audio_sample_rate=16000 \
  --n_mels=128
```

**Key Parameters**:
- `--audioset_labels_csv`: Enables AudioSet multi-label mode
- `--clip_duration=8.0`: Audio clip duration (MBT uses 8s for AudioSet)
- `--rgb_duration=3.0`: RGB clip duration (MBT uses 3s, ~75 frames at 25fps)

### Step 3: Verify TFRecords

```python
import tensorflow as tf

# Load one example
dataset = tf.data.TFRecordDataset('Datasets/audioset_eval_fullset/train-00000-of-00100.tfrecord')
for raw_record in dataset.take(1):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    
    # Check multi-hot label
    multi_hot = example.context.feature['clip/label/multi_hot'].float_list.value
    num_active = example.context.feature['clip/label/num_active'].int64_list.value[0]
    
    print(f"Multi-hot shape: {len(multi_hot)}")  # Should be 527
    print(f"Active classes: {num_active}")
    print(f"Non-zero indices: {[i for i, v in enumerate(multi_hot) if v > 0]}")
```

## AudioSet Label Structure

### audioset_labels.csv Format

```csv
index,mid,display_name
0,/m/09x0r,"Speech"
1,/m/05zppz,"Male speech, man speaking"
...
111,/m/015p6,"Bird"
...
526,/m/07rgkc5,"Zipper (clothing)"
```

### Example Multi-Label Conversion

**Input label string**: `"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"`

**Lookup in audioset_labels.csv**:
- `/m/068hy` → index 137 (example)
- `/m/07q6cd_` → index 298 (example)
- `/m/0bt9lr` → index 412 (example)
- `/m/0jbk` → index 56 (example)

**Multi-hot array**:
```python
[0, 0, ..., 1, ..., 0, 1, ..., 1, ..., 1, ..., 0]  # 527 elements
#           ^56      ^137     ^298     ^412
```

## Compatibility

### Backward Compatibility
- Single-label datasets (VGGSound): Use without `--audioset_labels_csv` flag
- Multi-label datasets (AudioSet): Use with `--audioset_labels_csv` flag

### MBT Checkpoint Compatibility
- Pretrained MBT checkpoints expect labels in range [0, 526]
- New pipeline ensures correct mapping to AudioSet's official indices
- Multi-hot encoding matches AudioSet's multi-label nature

## Next Steps

After preprocessing:

1. **Extract Activations**:
```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/audioset_classification.py \
  --checkpoint_dir=mbt_base \
  --test_data_dir=Datasets/audioset_eval_fullset \
  --output_dir=audioset_analysis_corrected \
  --num_samples=20371
```

2. **Compute RDMs**:
```bash
python compute_rdm.py \
  --activation_dir=audioset_analysis_corrected \
  --output_dir=ARDM_audioset_corrected \
  --distance_metric=correlation \
  --average_by_class=True \
  --label_mapping_file=Video_csvs/audioset_labels.csv
```

## Troubleshooting

### "Label not found in mid_to_index"
- Check that `audioset_labels.csv` contains all MIDs from your data
- AudioSet ontology may have been updated; use matching versions

### "No frames extracted"
- Verify video files exist and are not corrupted
- Check `--rgb_duration` doesn't exceed available video length
- Enable `--check_duration` flag for validation

### TFRecords missing multi_hot feature
- Ensure `--audioset_labels_csv` flag is provided
- Check that `mid_to_index` is passed to `process_video_entry()`
- Verify `generate_audiovisual_from_file.py` has Union import

## References

- AudioSet: https://research.google.com/audioset/
- AudioSet Ontology: https://research.google.com/audioset/ontology/index.html
- MBT Paper: "Multimodal Bottleneck Transformer"
