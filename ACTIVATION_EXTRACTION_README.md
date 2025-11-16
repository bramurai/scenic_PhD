# Neural Activation and Attention Analysis for MBT

This directory contains tools for extracting and analyzing neural activations and attention weights from the trained MBT model.

## Files Created

1. **`Inference_config.py`** - Configuration for activation extraction
   - Points to the 9 test samples in `Audioset_test/`
   - Disables training augmentations
   - Sets batch_size=1 for detailed per-sample analysis

2. **`activation_extractor.py`** - Module for capturing activations
   - `ActivationExtractor` class wraps the model
   - Captures intermediate layer outputs
   - Saves attention weights

3. **`extract_activations.py`** - Main extraction script
   - Loads trained checkpoint
   - Processes test samples
   - Saves activations to disk

4. **`run_inference_analysis.py`** - Alternative using Scenic framework
   - Integrates with existing Scenic infrastructure
   - Uses modified trainer for inference

## Quick Start

### Step 1: Extract Activations

```bash
conda activate scenic_phd

# Simple extraction script
python extract_activations.py \
  --checkpoint_path=mbt_base \
  --test_data_path=Audioset_test/data-00000-of-00001.tfrecord \
  --output_dir=activations_output \
  --num_samples=9
```

This will create:
- `activations_output/activations_sample_0000.npz` through `activations_sample_0008.npz`
- `activations_output/all_predictions.npz` - combined predictions
- `activations_output/config.pkl` - configuration used

### Step 2: Analyze Activations

```python
import numpy as np
import matplotlib.pyplot as plt

# Load activations for sample 0
data = np.load('activations_output/activations_sample_0000.npz')

# Check what's available
print("Available keys:", list(data.keys()))

# Load logits
logits = data['logits']
print("Logits shape:", logits.shape)

# Load layer activations (if captured)
for key in data.keys():
    if key.startswith('activation_'):
        layer_name = key.replace('activation_', '')
        activation = data[key]
        print(f"{layer_name}: {activation.shape}")
```

## Understanding the Outputs

### Activation Files

Each `.npz` file contains:
- `logits`: Final model predictions (shape: [1, num_classes])
- `activation_*`: Intermediate layer outputs
  - Naming follows the model hierarchy
  - Shapes vary by layer (attention, MLP, etc.)

### Layer Hierarchy

The MBT model has the following structure:
```
Input Embeddings (RGB + Spectrogram)
  ↓
Encoder Layers (0-7): Pre-fusion
  - Multi-head Attention
  - MLP
  ↓
Bottleneck Tokens (injected at layer 8)
  ↓
Fusion Layers (8-11): Cross-modal attention
  - Bottleneck Attention
  - Multi-head Attention
  - MLP
  ↓
GAP Classifier
  ↓
Logits (527 classes)
```

## Advanced Usage

### Extracting Specific Layer Activations

To capture specific layers, modify `extract_activations_with_intermediate_capture`:

```python
# Capture only attention layers
output, collected = model.flax_model.apply(
    variables,
    inputs,
    train=False,
    mutable=False,
    capture_intermediates=lambda module, method_name: (
        'attention' in module.name.lower() and method_name == '__call__'
    )
)
```

### Extracting Attention Weights

Attention weights require modifying the model to return them explicitly. Create a custom attention layer:

```python
class AttentionWithWeights(nn.Module):
    """Attention layer that returns weights."""
    
    def __call__(self, x, return_weights=False):
        # ... attention computation ...
        
        if return_weights:
            return output, attention_weights
        return output
```

### Visualizing Activations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load activations
data = np.load('activations_output/activations_sample_0000.npz')

# Find attention layer outputs
attention_layers = [k for k in data.keys() if 'attention' in k.lower()]

for layer_key in attention_layers:
    activation = data[layer_key]
    
    # Visualize first 10 channels
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < activation.shape[-1]:
            # Average over spatial dimensions if needed
            channel_data = activation[0, :, idx] if activation.ndim == 3 else activation[0, idx]
            ax.imshow(channel_data, cmap='viridis')
            ax.set_title(f'Channel {idx}')
            ax.axis('off')
    
    plt.suptitle(f'Activations: {layer_key}')
    plt.tight_layout()
    plt.savefig(f'visualizations/{layer_key}.png')
    plt.close()
```

## Troubleshooting

### Issue: "Checkpoint not found"
- Ensure `--checkpoint_path` points to your trained model directory
- Should contain files like `checkpoint_1`, `checkpoint`, etc.

### Issue: "Out of memory"
- Reduce `--num_samples` to process fewer samples
- Some layers produce very large activations

### Issue: "No activations captured"
- The intermediate capture might not work for all layer types
- Try modifying the `capture_intermediates` lambda function
- Check model implementation for mutable state

### Issue: "TFRecord parsing errors"
- Update the `load_test_data` function to match your TFRecord format
- Check the actual feature format in your TFRecords

## Next Steps

1. **Implement proper TFRecord parsing** - The current `load_test_data` is a placeholder
2. **Add attention weight extraction** - Requires modifying the attention layers
3. **Create visualization notebooks** - For exploring extracted activations
4. **Implement layer-wise analysis** - Compare activations across samples

## References

- Flax intermediate capture: https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html#flax.linen.capture_intermediates
- JAX debugging: https://jax.readthedocs.io/en/latest/debugging/index.html
