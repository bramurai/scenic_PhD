# MBT Inference Bug Fix

## Issue
When running `extract_activations_example.py` without a checkpoint (random initialization), the script crashed with:

```
flax.errors.ScopeParamShapeError: Initializer expected to generate shape (16, 16, 3, 768) 
but got shape (16, 16, 768, 768) instead for parameter "kernel" in "/embedding_spectrogram".
```

## Root Cause

The MBT model **mutates the input dictionary in-place** during the forward pass. When we called:

```python
# Initialize model
variables = model_instance.flax_model.init(rng, test_input, train=False)

# Run forward pass - uses SAME test_input!
outputs = model_instance.flax_model.apply(variables, test_input, train=False)
```

The sequence of events was:
1. `init()` calls the model, which transforms `test_input['spectrogram']` from shape `(1, 800, 128, 3)` to `(1, 400, 768)`
2. The input dict is modified in-place: `test_input['spectrogram']` now has shape `(1, 400, 768)`
3. `apply()` is called with the SAME (now modified) `test_input`
4. Model tries to embed already-embedded data, causing shape mismatch

This is particularly insidious because:
- The model's `temporal_encode()` function transforms spectrograms: `(batch, 800, 128, 3)` → `(batch, 400, 768)`
- Python dicts are mutable, so changes persist across function calls
- JAX operations typically don't modify data in-place, so this behavior is unexpected

## Solution

Create a copy of the input dictionary before initialization:

```python
# BEFORE (broken):
variables = model_instance.flax_model.init(rng, test_input, train=False)
outputs = model_instance.flax_model.apply(variables, test_input, train=False)

# AFTER (fixed):
test_input_init = {k: v for k, v in test_input.items()}  # Create copy
variables = model_instance.flax_model.init(rng, test_input_init, train=False)
outputs = model_instance.flax_model.apply(variables, test_input, train=False)
```

Now each call gets its own fresh input data.

## Files Changed

- **`scenic/projects/mbt/extract_activations_example.py`** (line 240):
  - Added: `test_input_init = {k: v for k, v in test_input.items()}`
  - Changed: `init()` now uses `test_input_init` instead of `test_input`

## Debugging Process

1. **Initial error**: Shape mismatch suggesting 768 input channels instead of 3
2. **Added logging**: Discovered `embed_2d_patch()` was being called twice
3. **More logging**: Found `temporal_encode('spectrogram')` called twice:
   - First: input `(1, 800, 128, 3)` ✓
   - Second: input `(1, 400, 768)` ✗ (already embedded!)
4. **Root cause identified**: Input dict modified in-place during first call
5. **Fix applied**: Copy input dict before initialization

## Lessons Learned

- **JAX/Flax models can mutate input dicts**: Even though JAX arrays are immutable, Python dict containers are not
- **Always copy mutable inputs**: When calling the same model multiple times with potentially modified inputs
- **Watch for in-place modifications**: Especially in loops that process dictionary values

## Testing

The fix was verified by running:
```bash
python -m scenic.projects.mbt.extract_activations_example \
    --input_source=dummy \
    --output_dir=./test_run
```

**Expected output**: Successfully generates:
- `predictions.npy`: Model predictions (1, 527)
- `fused_features.npy`: Fused multimodal features
- `config.txt`: Configuration summary

## Related Issues

This same pattern could affect other parts of the codebase where models are called multiple times with the same input dict. Areas to watch:
- Model initialization in training loops
- Evaluation/testing with multiple forward passes
- Any code doing `init()` followed by `apply()` on the same inputs

## Prevention

When writing inference/evaluation code:
```python
# Good practice: Always copy inputs when calling model multiple times
for batch in dataset:
    # Make a copy if you'll reuse the batch
    batch_copy = {k: v for k, v in batch.items()}
    result = model(batch_copy)
```

Or better yet:
```python
# Best practice: Use different variables for different purposes
init_batch = get_batch()
inference_batch = get_batch()  # Get fresh batch for inference
```
