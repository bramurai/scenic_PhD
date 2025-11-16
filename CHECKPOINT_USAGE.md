# How to Use Multiple MBT Checkpoints

## Checkpoint Directory Structure

```
CheckPoints/
├── mbt_run1/              ← Training run 1
│   ├── checkpoint         ← Points to latest checkpoint
│   ├── checkpoint_1       ← Checkpoint at step 1
│   ├── checkpoint_1000    ← Checkpoint at step 1000
│   ├── checkpoint_5000    ← Checkpoint at step 5000
│   └── ...
├── mbt_run2/              ← Training run 2
│   ├── checkpoint
│   ├── checkpoint_1
│   └── ...
└── mbt_run3/              ← Training run 3
    ├── checkpoint
    └── ...
```

## How to Specify Checkpoints

### Point to a specific training run (latest checkpoint)

```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \     # ← Specific run directory
  --test_data_dir=... \
  --output_dir=activations_run1
```

### Point to a specific checkpoint step

```bash
python extract_mbt_activations.py \
  --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=5000 \                     # ← Specific step
  --test_data_dir=... \
  --output_dir=activations_run1_step5000
```

## Finding Available Checkpoints

```bash
# List all training runs
ls CheckPoints/

# See what checkpoints exist in a run
ls CheckPoints/mbt_run1/checkpoint_*

# See which checkpoint is "latest"
cat CheckPoints/mbt_run1/checkpoint
```

## Comparing Multiple Checkpoints

### Option 1: Different training runs

```bash
# Extract from run 1
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --output_dir=activations_run1 \
  ...

# Extract from run 2
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run2 \
  --output_dir=activations_run2 \
  ...

# Compare results
python compare_runs.py --run1=activations_run1 --run2=activations_run2
```

### Option 2: Different steps in same run

```bash
# Early training (step 1000)
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=1000 \
  --output_dir=activations_step1000 \
  ...

# Middle training (step 5000)
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=5000 \
  --output_dir=activations_step5000 \
  ...

# Final training (latest)
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --output_dir=activations_final \
  ...
```

### Option 3: Batch processing

```bash
# Process all runs
for run in CheckPoints/mbt_*; do
  run_name=$(basename $run)
  python extract_mbt_activations.py \
    --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
    --checkpoint_dir=$run \
    --test_data_dir=/media/labuta/.../test_tfrecords_local \
    --output_dir=activations_${run_name} \
    --num_samples=100
done
```

## Common Patterns

### 1. Extract from best checkpoint

If you know which checkpoint performed best (e.g., step 8000):

```bash
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=8000 \
  --test_data_dir=... \
  --output_dir=activations_best
```

### 2. Compare early vs late training

```bash
# Early (step 1000)
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=1000 \
  --output_dir=activations_early

# Late (step 10000)
python extract_mbt_activations.py \
  --checkpoint_dir=CheckPoints/mbt_run1 \
  --checkpoint_step=10000 \
  --output_dir=activations_late

# Compare how representations evolved
python analyze_activations.py --activation_dir=activations_early --output_dir=pca_early
python analyze_activations.py --activation_dir=activations_late --output_dir=pca_late
```

### 3. Extract from all checkpoints in a run

```bash
# Get all checkpoint steps
for ckpt in CheckPoints/mbt_run1/checkpoint_*; do
  step=$(basename $ckpt | sed 's/checkpoint_//')
  
  python extract_mbt_activations.py \
    --checkpoint_dir=CheckPoints/mbt_run1 \
    --checkpoint_step=$step \
    --test_data_dir=... \
    --output_dir=activations_step${step} \
    --num_samples=50  # Use fewer samples for each step
done
```

## Tips

1. **Use descriptive output directories**: Name them after the run and step
   - Good: `activations_run1_step5000`
   - Bad: `activations_output`

2. **Start with fewer samples**: Use `--num_samples=10` first to test
   - Then increase to 100 or more for full analysis

3. **Check checkpoint file**: Look at `CheckPoints/mbt_run1/checkpoint` to see what's available

4. **Save disk space**: Only extract from checkpoints you need
   - Each extraction can be several GB depending on `--num_samples`

5. **Organize results**: Keep activations from different checkpoints separate
   ```
   activations/
     run1_step1000/
     run1_step5000/
     run2_step1000/
     ...
   ```
