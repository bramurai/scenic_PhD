"""Grid search over inference parameters using Pass 1 mAP as metric.

Tests different combinations of:
- num_frames (8, 16, 32)
- spec_mean/stddev (different normalizations)
- fusion_layer (6, 8, 10)

Uses the fast Pass 1 (logits-only with pmap) to quickly evaluate each config.
"""
import subprocess
import os
import re
import itertools
from pathlib import Path

# Grid search parameters
GRID = {
    'temporal_window_seconds': [1.0, 2.0, 4.0, 8.0],  # Paper: sample 8 frames over window of length t
    'spec_mean': [0.0, 1.102],  # 0.0 = no mean subtraction, 1.102 = from config
    'spec_stddev': [1.0, 2.762],  # 1.0 = no scaling, 2.762 = from config
    'fusion_layer': [6, 8],  # Common choices in MBT configs
}

# Fixed from paper: "we sample 8 RGB frames"
NUM_FRAMES_FIXED = 8
FPS = 25  # AudioSet videos at 25fps

# Fixed parameters
BASE_CONFIG = '/project/3026018.01/Models/MBT/scenic/projects/mbt/configs/audioset/Inference_config.py'
CHECKPOINT_DIR = '/project/3026018.01/Models/MBT/CheckPoints/MBT_AV'
TEST_DATA_DIR = '/project/3026018.01/Models/MBT/Datasets/audioset_eval'
LABELS_CSV = '/project/3026018.01/Models/MBT/Video_csvs/audioset_labels.csv'
OUTPUT_BASE = '/project/3026018.01/Models/MBT/grid_search_results'

# Create output directory
os.makedirs(OUTPUT_BASE, exist_ok=True)

def create_temp_config(temporal_window_seconds, spec_mean, spec_stddev, fusion_layer, config_path):
    """Create a temporary config file with modified parameters."""
    # Read base config
    with open(BASE_CONFIG, 'r') as f:
        config_content = f.read()
    
    # Calculate num_frames and stride from temporal window
    # Paper: "sample 8 RGB frames over the sampling window of length t with uniform stride (t × 25)/8"
    num_frames = NUM_FRAMES_FIXED  # Always 8 for AudioSet
    stride = int((temporal_window_seconds * FPS) / num_frames)  # stride = (t × 25) / 8
    
    # 1. num_frames (always 8)
    config_content = re.sub(
        r'config\.dataset_configs\.num_frames = \d+',
        f'config.dataset_configs.num_frames = {num_frames}',
        config_content
    )
    
    # 2. stride (calculated from temporal window)
    config_content = re.sub(
        r'config\.dataset_configs\.stride = \d+',
        f'config.dataset_configs.stride = {stride}',
        config_content
    )
    
    # 3. spec_mean
    config_content = re.sub(
        r'config\.dataset_configs\.spec_mean = [\d\.]+',
        f'config.dataset_configs.spec_mean = {spec_mean}',
        config_content
    )
    
    # 4. spec_stddev
    config_content = re.sub(
        r'config\.dataset_configs\.spec_stddev = [\d\.]+',
        f'config.dataset_configs.spec_stddev = {spec_stddev}',
        config_content
    )
    
    # 5. fusion_layer
    config_content = re.sub(
        r'config\.model\.fusion_layer = \d+',
        f'config.model.fusion_layer = {fusion_layer}',
        config_content
    )
    
    # Write to temp file
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path, num_frames, stride

def run_pass1_only(config_path, output_dir):
    """Run only Pass 1 (logits/mAP) to quickly evaluate config."""
    cmd = [
        'python3', 'extract_mbt_activations_class_averaged.py',
        f'--config={config_path}',
        f'--checkpoint_dir={CHECKPOINT_DIR}',
        f'--test_data_dir={TEST_DATA_DIR}',
        f'--output_dir={output_dir}',
        f'--audioset_labels_csv={LABELS_CSV}',
        '--nosave_activations',  # Skip activations for speed
        '--save_logits',
        '--compute_map',
        '--two_pass=False',  # Only do Pass 1
        '--batch_size=4',
        '--num_samples=80',  # Use all 80 test samples
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def extract_map_from_output(output_text):
    """Extract mAP score from script output."""
    # Look for "[Pass 1] mAP: 0.XXXX"
    match = re.search(r'\[Pass 1\] mAP: ([\d\.]+)', output_text)
    if match:
        return float(match.group(1))
    return None

# Run grid search
results = []
total_configs = len(list(itertools.product(*GRID.values())))
config_num = 0

print(f"Starting grid search over {total_configs} configurations...")
print(f"Parameters: {GRID}")
print("="*70)

for temporal_window_seconds, spec_mean, spec_stddev, fusion_layer in itertools.product(*GRID.values()):
    config_num += 1
    config_name = f't{temporal_window_seconds}s_sm{spec_mean}_ss{spec_stddev}_fl{fusion_layer}'
    print(f"\n[{config_num}/{total_configs}] Testing: {config_name}")
    
    # Create temp config
    temp_config = os.path.join(OUTPUT_BASE, f'config_{config_name}.py')
    config_path, num_frames, stride = create_temp_config(
        temporal_window_seconds, spec_mean, spec_stddev, fusion_layer, temp_config
    )
    print(f"  num_frames={num_frames}, stride={stride} (window={temporal_window_seconds}s)")
    
    # Create output directory for this config
    output_dir = os.path.join(OUTPUT_BASE, config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Pass 1 only
    print(f"  Running Pass 1 (logits/mAP)...")
    result = run_pass1_only(config_path, output_dir)
    
    # Extract mAP
    map_score = extract_map_from_output(result.stderr + result.stdout)
    
    if map_score is not None:
        print(f"  ✓ mAP: {map_score:.4f}")
        results.append({
            'temporal_window_seconds': temporal_window_seconds,
            'num_frames': num_frames,
            'stride': stride,
            'spec_mean': spec_mean,
            'spec_stddev': spec_stddev,
            'fusion_layer': fusion_layer,
            'map': map_score,
            'config_name': config_name
        })
    else:
        print(f"  ✗ Failed to extract mAP")
        # Save error log
        with open(os.path.join(output_dir, 'error.log'), 'w') as f:
            f.write(result.stderr)
        results.append({
            'temporal_window_seconds': temporal_window_seconds,
            'num_frames': num_frames,
            'stride': stride,
            'spec_mean': spec_mean,
            'spec_stddev': spec_stddev,
            'fusion_layer': fusion_layer,
            'map': 0.0,
            'config_name': config_name,
            'error': True
        })

# Sort results by mAP
results.sort(key=lambda x: x.get('map', 0), reverse=True)

# Print summary
print("\n" + "="*70)
print("GRID SEARCH RESULTS (sorted by mAP):")
print("="*70)
for i, res in enumerate(results[:10], 1):  # Top 10
    print(f"{i}. mAP={res['map']:.4f} | t={res['temporal_window_seconds']}s "
          f"(nf={res['num_frames']}, stride={res['stride']}), fl={res['fusion_layer']}, "
          f"sm={res['spec_mean']}, ss={res['spec_stddev']}")

# Save full results
import json
results_path = os.path.join(OUTPUT_BASE, 'grid_search_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nFull results saved to: {results_path}")
print(f"\nBest configuration:")
best = results[0]
print(f"  temporal_window = {best['temporal_window_seconds']} seconds")
print(f"  num_frames = {best['num_frames']} (fixed from paper)")
print(f"  stride = {best['stride']} (calculated: {best['temporal_window_seconds']}×25/{best['num_frames']})")
print(f"  fusion_layer = {best['fusion_layer']}")
print(f"  spec_mean = {best['spec_mean']}")
print(f"  spec_stddev = {best['spec_stddev']}")
print(f"  mAP = {best['map']:.4f}")
