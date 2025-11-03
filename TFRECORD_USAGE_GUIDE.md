# VGGSound TFRecord Usage Guide

## Understanding the 2-Shard TFRecord Structure

Each batch archive contains **2 TFRecord shards**:
```
batch_00000/
├── data-00000-of-00002.tfrecord  # Shard 0 (first half of batch)
└── data-00001-of-00002.tfrecord  # Shard 1 (second half of batch)
```

**Why 2 shards?**
- Enables parallel reading during training
- Better distribution across GPUs/workers
- Prevents memory issues with large batches

**Important:** You don't need to manually merge shards! TensorFlow automatically reads both shards in parallel.

---

## How to Use TFRecords for Training

### Step 1: Extract All Archives

Extract all downloaded `*.tar.gz` files:

```powershell
# For test set
cd test_tfrecords_local
Get-ChildItem *.tar.gz | ForEach-Object { tar -xzf $_.Name }

# For train set (when ready)
cd ..\train_tfrecords_local
Get-ChildItem *.tar.gz | ForEach-Object { tar -xzf $_.Name }
```

**Result:** Creates `batch_*/` directories with TFRecord shards inside.

---

### Step 2: Update Config File

Edit `scenic/projects/mbt/configs/audioset/vggsound_base.py`:

```python
config.dataset_configs.base_dir = 'C:/Users/bravhee/Uta_PhD/scenic_PhD/scenic_PhD'

config.dataset_configs.tables = {
    # Glob pattern automatically finds ALL TFRecord shards in ALL batches
    'train': 'train_tfrecords_local/batch_*/data-*-of-*.tfrecord',
    'validation': 'test_tfrecords_local/batch_*/data-*-of-*.tfrecord',
    'test': 'test_tfrecords_local/batch_*/data-*-of-*.tfrecord',
}

config.dataset_configs.examples_per_subset = {
    'train': 183971,   # Your actual train CSV count
    'validation': 15496,  # Your actual test CSV count
    'test': 15496
}

config.dataset_configs.num_classes = 309  # VGGSound has 309 classes
```

---

### Step 3: Train the Model

```bash
cd scenic

python -m scenic.projects.mbt.main \
  --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
  --workdir=/path/to/output/checkpoints
```

---

## Data Flow Explanation

### What's in Each TFRecord?

Each shard contains:
- **RGB frames**: 32 frames × 224×224×3 (video data)
- **Spectrogram**: 100×128 (audio data)
- **Label**: One-hot encoded vector (309 classes for VGGSound)

### How TensorFlow Reads the Data

1. **Glob pattern matching**: `batch_*/data-*-of-*.tfrecord` finds all shards
2. **Automatic parallelization**: TensorFlow reads both shards (`00000` and `00001`) simultaneously
3. **Shuffling**: Data is shuffled across and within shards
4. **Batching**: Assembles batches according to `config.batch_size`
5. **Distribution**: Distributes batches across GPUs/workers

### File Structure After Extraction

```
scenic_PhD/
├── train_tfrecords_local/
│   ├── batch_00000/
│   │   ├── data-00000-of-00002.tfrecord
│   │   └── data-00001-of-00002.tfrecord
│   ├── batch_00200/
│   │   ├── data-00000-of-00002.tfrecord
│   │   └── data-00001-of-00002.tfrecord
│   └── ...
└── test_tfrecords_local/
    ├── batch_00000/
    │   ├── data-00000-of-00002.tfrecord
    │   └── data-00001-of-00002.tfrecord
    └── ...
```

---

## Key Points

✅ **No manual merging needed** - TensorFlow handles the 2-shard structure automatically

✅ **Keep directory structure** - The `batch_*/` organization is fine, just use glob patterns

✅ **Glob patterns are powerful** - `data-*-of-*.tfrecord` matches all shards automatically

✅ **Both shards are required** - Don't delete either shard; they work together

✅ **Update dataset sizes** - Use actual counts from your CSVs (183971 train, 15496 test)

✅ **Update num_classes** - VGGSound has 309 classes, not 527 like AudioSet

---

## Alternative: Flattened Structure (Optional)

If you prefer a flatter structure, you can reorganize:

```python
# Create train/ and test/ directories with all TFRecords
train/
├── data-00000-of-XXXXX.tfrecord
├── data-00001-of-XXXXX.tfrecord
├── data-00002-of-XXXXX.tfrecord
└── ...

# Update config:
config.dataset_configs.tables = {
    'train': 'train/data-*-of-*.tfrecord',
    'validation': 'test/data-*-of-*.tfrecord',
}
```

But this is **not necessary** - the batch structure works perfectly fine!

---

## Troubleshooting

**Q: Model says "No matching files found"**
- Check `base_dir` path is correct (use absolute path)
- Verify TFRecords are extracted (not still in `.tar.gz`)
- Ensure glob pattern matches your directory structure

**Q: Training is slow**
- Check you have both shards per batch (enables parallelization)
- Verify `prefetch_to_device = 2` in config
- Consider using more workers if on multi-GPU setup

**Q: Out of memory errors**
- Reduce `batch_size` in config (default is 64)
- Check GPU memory usage
- Reduce `num_frames` if needed (default is 32)
