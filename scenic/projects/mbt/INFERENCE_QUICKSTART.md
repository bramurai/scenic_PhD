# MBT Inference Quick Start Guide

## What is the Input Data?

The `extract_activations_example.py` script processes **multimodal input** consisting of:

1. **RGB Video**: Shape `(batch_size, 32, 224, 224, 3)`
   - 32 frames per video clip
   - 224×224 pixel resolution
   - 3 color channels (RGB)

2. **Audio Spectrogram**: Shape `(batch_size, 800, 128, 3)`
   - 800 time bins
   - 128 frequency bins  
   - 3 channels (log-mel spectrogram features)

## Three Input Options

### Option 1: Dummy Data (Default) 🔧
**Use case**: Architecture analysis, testing model structure without real data

```bash
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dummy \
    --output_dir=./analysis_dummy
```

**What happens**: Creates all-zero tensors for both modalities. Useful for:
- Understanding model architecture
- Checking activation shapes at each layer
- Testing inference pipeline without real data
- Quick debugging

⚠️ **Limitation**: Results won't reflect real activation patterns since inputs are zeros.

---

### Option 2: Dataset Loading 📊
**Use case**: Analyzing real samples from AudioSet or custom TFRecord dataset

```bash
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dataset \
    --output_dir=./analysis_dataset \
    --checkpoint_path=./Pretrained_Models/mbtb32_as-500k_rgb-spec/checkpoint
```

**What happens**: Loads actual samples from the dataset configured in your config file. Requires:
- TFRecord dataset files in the path specified by config
- Dataset preprocessing pipeline set up correctly
- Sufficient disk I/O for loading samples

✅ **Advantage**: Gets real activation patterns from training/validation data.

---

### Option 3: Custom Numpy Files 📁
**Use case**: Analyzing specific video clips you've preprocessed

First, prepare your data:
```python
import numpy as np

# Your video data (32 frames, 224×224, RGB)
rgb_data = np.random.randn(4, 32, 224, 224, 3).astype(np.float32)
np.save('my_rgb_samples.npy', rgb_data)

# Your audio spectrogram
spec_data = np.random.randn(4, 800, 128, 3).astype(np.float32)
np.save('my_spec_samples.npy', spec_data)
```

Then run inference:
```bash
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=file \
    --rgb_file=my_rgb_samples.npy \
    --spec_file=my_spec_samples.npy \
    --output_dir=./analysis_custom \
    --checkpoint_path=./Pretrained_Models/mbtb32_as-500k_rgb-spec/checkpoint
```

✅ **Advantage**: Full control over input samples, ideal for specific analysis tasks.

---

## Complete Command-Line Options

```bash
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dummy|dataset|file \   # Input data source
    --rgb_file=/path/to/rgb.npy \         # Required if input_source=file
    --spec_file=/path/to/spec.npy \       # Required if input_source=file
    --checkpoint_path=/path/to/ckpt \     # Optional: pretrained model
    --output_dir=./my_analysis            # Where to save results
```

## Output Files

The script saves:

1. **`final_output.npy`**: Model predictions (batch_size, 527 classes)
2. **`activation_stats.json`**: Statistics for each layer:
   - Mean, std, min, max activation values
   - Sparsity (% of near-zero activations)
   - Shape information
3. **`modality_features.npy`**: Separate RGB and spectrogram features

## Memory Requirements

| Batch Size | Memory (float32) | Memory (float16) |
|------------|------------------|------------------|
| 1          | ~0.5 GB          | ~0.3 GB          |
| 4          | ~0.8 GB          | ~0.5 GB          |
| 8          | ~1.2 GB          | ~0.7 GB          |
| 16         | ~2.0 GB          | ~1.2 GB          |

Your system: **7.4 GB RAM** → Safe to use batch_size=8 or 16

## Example Workflow

### 1. Test Architecture (No Checkpoint Needed)
```bash
# Quick test with dummy data
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dummy \
    --output_dir=./test_run
```

**Expected output**: Model structure analysis, random activations

---

### 2. Analyze Real Data with Pretrained Model
```bash
# Load from dataset with pretrained checkpoint
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dataset \
    --checkpoint_path=./Pretrained_Models/mbtb32_as-500k_rgb-spec/checkpoint \
    --output_dir=./real_analysis
```

**Expected output**: Meaningful activation patterns from trained model

---

### 3. Custom Sample Analysis
```bash
# Prepare your samples first
python prepare_my_samples.py  # Your preprocessing script

# Run inference
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=file \
    --rgb_file=my_video.npy \
    --spec_file=my_audio.npy \
    --checkpoint_path=./Pretrained_Models/mbtb32_as-500k_rgb-spec/checkpoint \
    --output_dir=./my_sample_analysis
```

---

## Troubleshooting

### "ScopeParamShapeError: Initializer expected to generate shape..."
- **Cause**: Model mutates input dict in-place during forward pass
- **Solution**: Already fixed! The script now creates a copy of the input for initialization
- **Technical Details**: JAX/Flax `init()` and `apply()` both call the model, and the MBT model modifies the input dictionary's values during execution. Using the same dict twice causes shape mismatches.

### "No checkpoint provided, using random initialization"
- **Cause**: Didn't specify `--checkpoint_path`
- **Effect**: Model uses random weights (not trained)
- **Solution**: Add `--checkpoint_path=/path/to/checkpoint` or accept random initialization for architecture testing

### "Must specify --rgb_file and --spec_file when using input_source=file"
- **Cause**: Selected file input but didn't provide file paths
- **Solution**: Add both `--rgb_file` and `--spec_file` flags

### "FileNotFoundError: TFRecord dataset not found"
- **Cause**: Dataset path in config doesn't match your system
- **Effect**: Cannot load real data
- **Solution**: Use `--dataset_path` flag:
  ```bash
  python -m scenic.projects.mbt.extract_activations_example \
      --input_source=dataset \
      --dataset_path=/path/to/your/dataset \
      --checkpoint_path=./checkpoint
  ```

### "Feature list 'melspec/feature/floats' is required but could not be found"
- **Cause**: Your TFRecord file doesn't have the expected AudioSet format with mel-spectrograms
- **Effect**: Cannot load dataset
- **Solution**: Either:
  1. Use dummy data: `--input_source=dummy` (no real data needed)
  2. Use numpy files: `--input_source=file --rgb_file=X --spec_file=Y`
  3. Recreate TFRecord with proper AudioSet preprocessing (includes mel-spectrogram extraction)

### Out of Memory (OOM)
- **Cause**: Batch size too large for your RAM
- **Solution**: Modify `modify_config_for_activation_extraction()` to reduce batch_size:
  ```python
  config.batch_size = 1  # Minimum batch size
  ```

---

## Next Steps

- **Want layer-by-layer features?** See `ACTIVATION_EXTRACTION.md` for model modifications
- **Want attention weights?** Check the attention extraction examples in the docs
- **Want to train your own model?** Reduce batch_size to 4-8 in training config
