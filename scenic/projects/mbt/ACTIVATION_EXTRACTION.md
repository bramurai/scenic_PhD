# MBT Activation Extraction Guide

This guide explains how to extract activation patterns from the Multimodal Bottleneck Transformer (MBT) model for analysis.

## Overview

The MBT model has several levels where you can extract activations:

1. **Patch Embeddings** - After temporal encoding, before transformer
2. **Transformer Layer Outputs** - After each encoder block
3. **Bottleneck Tokens** - Cross-modal fusion tokens (if enabled)
4. **Attention Weights** - Self-attention and cross-attention patterns
5. **Pre-classifier Features** - Final pooled representation before classification
6. **Final Predictions** - Class logits/probabilities

## Quick Start

### 1. Basic Inference (No Activation Extraction)

```bash
cd /mnt/c/Users/bramh/PhD/scenic_PhD

conda activate scenic_mbt

python -m scenic.projects.mbt.extract_activations_example \
  --checkpoint_path=/mnt/c/Users/bramh/PhD/scenic_PhD/Pretrained_Models/mbtb32_as-500k_rgb-spec \
  --output_dir=./activation_analysis \
  --batch_size=1
```

### 2. Extract with Float16 (Memory Efficient)

```bash
python -m scenic.projects.mbt.extract_activations_example \
  --checkpoint_path=/mnt/c/Users/bramh/PhD/scenic_PhD/Pretrained_Models/mbtb32_as-500k_rgb-spec \
  --output_dir=./activation_analysis \
  --batch_size=1 \
  --use_float16
```

### 3. Run Without Checkpoint (Architecture Analysis)

```bash
python -m scenic.projects.mbt.extract_activations_example \
  --output_dir=./architecture_analysis \
  --batch_size=1
```

## Configuration Modifications for Activation Extraction

### In `balanced_audioset_base.py` or programmatically:

```python
# 1. Disable stochastic operations (critical!)
config.model.dropout_rate = 0.0
config.model.attention_dropout_rate = 0.0
config.model.stochastic_droplayer_rate = 0.0

# 2. Disable data augmentation
config.dataset_configs.spec_augment = False
config.dataset_configs.augmentation_params.do_color_augment = False

# 3. Set batch size (1 for detailed per-sample analysis)
config.batch_size = 1

# 4. Enable intermediate outputs
config.model.return_preclassifier = True   # Get all token embeddings
config.model.return_prelogits = True       # Get pre-classification features

# 5. Optional: use float16 for memory efficiency
config.model_dtype_str = 'float16'
```

## What You Can Extract (Current Implementation)

### ✅ Available Now:

1. **Final Predictions**
   - Shape: `(batch_size, num_classes)`
   - Location: Model output
   - Use: Classification results

2. **Pre-logits Features** (if `return_prelogits=True`)
   - Shape: `(batch_size, hidden_size)` for GAP/GMP
   - Location: Before final classification layer
   - Use: Feature representations

3. **All Token Embeddings** (if `return_preclassifier=True`)
   - Shape: `(batch_size, num_tokens, hidden_size)`
   - Location: After encoder, before pooling
   - Use: Spatial/temporal token analysis

4. **Modality-Specific Outputs** (during training)
   - Separate predictions for RGB and spectrogram
   - Only available when `train=True` and model returns dict

### ⚠️ Requires Model Modification:

The following require modifying the model code to return intermediate values:

1. **Patch Embeddings**
   - Modify `temporal_encode()` to return embeddings
   - Location: After Conv layers, before transformer

2. **Layer-by-Layer Activations**
   - Modify `Encoder` class to collect outputs from each layer
   - Create a list/dict of layer outputs

3. **Bottleneck Tokens**
   - Modify `Encoder` to return `bottleneck` variable
   - Shape: `(batch_size, n_bottlenecks, hidden_size)`

4. **Attention Weights**
   - Modify `EncoderBlock` to return attention matrices
   - Shape: `(batch_size, num_heads, num_tokens, num_tokens)`

## Model Modification Examples

### Example 1: Extract Layer-wise Activations

Modify `model.py` - `Encoder` class:

```python
@nn.compact
def __call__(self, x: Dict[str, Any], bottleneck: jnp.ndarray, *, train: bool):
    # ... existing code ...
    
    layer_outputs = []  # ← Add this
    
    for lyr in range(self.num_layers):
        # ... existing layer processing ...
        
        # After processing each layer, save output
        layer_outputs.append({
            'layer': lyr,
            'rgb': x['rgb'] if 'rgb' in x else None,
            'spectrogram': x['spectrogram'] if 'spectrogram' in x else None,
        })
    
    # Return both final output and intermediate layers
    return encoded, layer_outputs
```

### Example 2: Extract Attention Weights

Modify `model.py` - `EncoderBlock` class:

```python
@nn.compact
def __call__(self, inputs: jnp.ndarray, deterministic: bool, return_attention: bool = False):
    # Attention block
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    
    # Modify attention call to return weights
    x, attention_weights = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        deterministic=deterministic
    )(x, x, deterministic=deterministic, return_attention=return_attention)
    
    # ... rest of the code ...
    
    if return_attention:
        return output, attention_weights
    return output
```

### Example 3: Extract Bottleneck Tokens

Modify `model.py` - `MBT` class `__call__` method:

```python
# After encoder
x, bottleneck_final = Encoder(...)(x, bottleneck, train=train)

# Return bottleneck with output
if self.return_preclassifier:
    return {'tokens': x, 'bottleneck': bottleneck_final}
```

## Memory Requirements

### Inference (Forward Pass Only)

| Configuration | Memory Usage | Recommended Batch Size |
|--------------|--------------|----------------------|
| Float32, batch=1 | ~1.0 GB | 1-8 |
| Float32, batch=8 | ~1.6 GB | 8-16 |
| Float16, batch=1 | ~0.6 GB | 1-16 |
| Float16, batch=16 | ~1.2 GB | 16-32 |

**Your System**: 7.4 GB total → Safe to use batch=8-16 with float32

### With Activation Storage

Add ~500 MB per batch for storing all layer activations in memory.

## Output Files

After running the extraction script, you'll get:

```
activation_analysis/
├── predictions.npy              # Final model predictions
├── rgb_features.npy            # RGB-specific features (if available)
├── spectrogram_features.npy   # Spectrogram features (if available)
├── fused_features.npy         # Fused multimodal features
├── activation_stats.npy       # Statistics for each layer
└── config.txt                 # Model configuration used
```

## Analysis Examples

### Load and Analyze Saved Activations

```python
import numpy as np

# Load predictions
predictions = np.load('activation_analysis/predictions.npy')
print(f"Prediction shape: {predictions.shape}")

# Load modality features
rgb_features = np.load('activation_analysis/rgb_features.npy')
spec_features = np.load('activation_analysis/spectrogram_features.npy')

# Compare modality representations
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(rgb_features, spec_features)
print(f"Cross-modal similarity: {similarity}")

# Load activation statistics
stats = np.load('activation_analysis/activation_stats.npy', allow_pickle=True).item()
for layer_name, layer_stats in stats.items():
    print(f"{layer_name}: mean={layer_stats['mean']:.3f}, std={layer_stats['std']:.3f}")
```

### Visualize Activation Patterns

```python
import matplotlib.pyplot as plt

# Plot prediction distribution
plt.figure(figsize=(10, 4))
plt.bar(range(len(predictions[0])), predictions[0])
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Class Predictions')
plt.savefig('predictions.png')

# Plot activation statistics across layers
layer_means = [stats[f'layer_{i}']['mean'] for i in range(12)]
plt.figure(figsize=(10, 4))
plt.plot(layer_means, marker='o')
plt.xlabel('Layer')
plt.ylabel('Mean Activation')
plt.title('Activation Magnitude Across Layers')
plt.savefig('layer_activations.png')
```

## Advanced: Custom Activation Extraction

For more control, use the inference module directly:

```python
from scenic.projects.mbt import inference
from scenic.projects.mbt.configs.audioset import balanced_audioset_base

# 1. Prepare config
config = balanced_audioset_base.get_config()
config = inference.prepare_inference_config('path/to/config.py')

# 2. Build dataset metadata
dataset_meta_data = inference.build_dataset_metadata(config)

# 3. Create input
input_data = {
    'rgb': jnp.zeros((1, 32, 224, 224, 3)),
    'spectrogram': jnp.zeros((1, 800, 128, 3))
}

# 4. Run inference
results = inference.run_inference(
    config,
    checkpoint_path='path/to/checkpoint',
    input_data=input_data,
    extract_intermediates=True
)

# 5. Access results
predictions = results['outputs']
if 'intermediates' in results:
    intermediates = results['intermediates']
```

## Key Points for Activation Analysis

1. **Always disable stochastic operations**: Set all dropout rates to 0
2. **Use batch_size=1** for detailed per-sample analysis
3. **Consider float16** for memory efficiency (small accuracy impact)
4. **Model returns dict during training**: Can get modality-specific outputs
5. **Attention extraction requires modification**: Not available out-of-the-box
6. **Bottleneck tokens are interesting**: They mediate cross-modal fusion

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the right directory
cd /mnt/c/Users/bramh/PhD/scenic_PhD

# Activate conda environment
conda activate scenic_mbt

# Check Python can find scenic
python -c "import scenic; print(scenic.__file__)"
```

### Out of memory
```bash
# Reduce batch size
--batch_size=1

# Use float16
--use_float16

# Or both
--batch_size=1 --use_float16
```

### Checkpoint loading fails
```bash
# Make sure path is correct
--checkpoint_path=/full/path/to/checkpoint

# Or run without checkpoint to analyze architecture
python -m scenic.projects.mbt.extract_activations_example \
  --output_dir=./analysis
```

## Further Reading

- Flax documentation on intermediate outputs: https://flax.readthedocs.io/
- JAX visualization tools: https://github.com/google/jax/tree/main/jax/experimental
- MBT paper: https://arxiv.org/abs/2107.00135
