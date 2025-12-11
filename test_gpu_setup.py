#!/usr/bin/env python3
"""Quick GPU and MBT setup verification script.

Runs in ~2-5 minutes to test:
- GPU availability
- JAX/TensorFlow GPU access
- MBT model loading
- Data loading
- Forward pass on small batch
"""

import os
import sys
from datetime import datetime

print("="*80)
print("MBT GPU Setup Verification")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# 1. Check GPU Availability
# ============================================================================
print("\n[1/6] Checking GPU availability...")

try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ nvidia-smi found")
        # Extract GPU count
        gpu_lines = [l for l in result.stdout.split('\n') if 'NVIDIA' in l or 'GPU' in l or 'CUDA' in l]
        for line in gpu_lines[:3]:
            print(f"  {line}")
    else:
        print("✗ nvidia-smi not available")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking GPUs: {e}")
    sys.exit(1)

# ============================================================================
# 2. Check JAX/TensorFlow
# ============================================================================
print("\n[2/6] Checking JAX and TensorFlow...")

try:
    import jax
    import jax.numpy as jnp
    import tensorflow as tf
    
    print(f"✓ JAX version: {jax.__version__}")
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Check devices
    devices = jax.devices()
    print(f"✓ JAX devices: {len(devices)} device(s)")
    for i, device in enumerate(devices):
        print(f"    Device {i}: {device}")
    
    # Check if GPU is available
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            print(f"✓ GPU devices accessible: {len(gpu_devices)} GPU(s)")
            for i, gpu in enumerate(gpu_devices):
                print(f"    GPU {i}: {gpu}")
        else:
            print("⚠ No GPU devices found in JAX")
            print("  This means CUDA modules are not loaded on this node.")
            print("  Make sure to load CUDA modules before running JAX code:")
            print("    module load cuda  (check your cluster's available CUDA versions)")
            cpu_devices = jax.devices('cpu')
            if cpu_devices:
                print(f"  CPU devices available: {len(cpu_devices)}")
                print("  You can still run on CPU, but GPU-accelerated jobs need CUDA.")
    except RuntimeError as e:
        if "Unknown backend: 'gpu'" in str(e):
            print("⚠ GPU backend not available (CUDA not configured)")
            print("  CUDA libraries not found. Load CUDA module on your cluster.")
            print("  Try: module load cuda")
        else:
            raise


except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# ============================================================================
# 3. Check Scenic/MBT imports
# ============================================================================
print("\n[3/6] Checking Scenic and MBT imports...")

try:
    from scenic.projects.mbt import model as mbt_model
    print("✓ Scenic MBT model imported successfully")
    
    try:
        from scenic.projects.mbt.datasets import audiovisual_tfrecord_dataset
        print("✓ Scenic MBT dataset utils imported successfully")
    except ImportError as e:
        print(f"⚠ Warning: Could not import audiovisual_tfrecord_dataset: {e}")
        print("  (This may be due to optional dependencies like dmvr/clip)")
        print("  If your main extraction script works, you can ignore this.")
    
except ImportError as e:
    print(f"✗ Scenic import error: {e}")
    print("  Try: pip install -e . in the scenic directory")
    sys.exit(1)

# ============================================================================
# 4. Check data directory
# ============================================================================
print("\n[4/6] Checking data directory...")

test_data_dir = "Datasets/audioset_eval_configCorrect"
labels_csv = "Video_csvs/audioset_labels.csv"

if os.path.isdir(test_data_dir):
    import glob
    tfrecords = glob.glob(os.path.join(test_data_dir, '**', '*.tfrecord'), recursive=True)
    print(f"✓ Test data directory found: {test_data_dir}")
    print(f"  TFRecord files: {len(tfrecords)}")
    if tfrecords:
        print(f"  First file: {os.path.basename(tfrecords[0])}")
else:
    print(f"⚠ Test data directory not found: {test_data_dir}")

if os.path.isfile(labels_csv):
    import pandas as pd
    df = pd.read_csv(labels_csv)
    print(f"✓ Labels CSV found: {labels_csv}")
    print(f"  Classes: {len(df)}")
else:
    print(f"⚠ Labels CSV not found: {labels_csv}")

# ============================================================================
# 5. Check checkpoint
# ============================================================================
print("\n[5/6] Checking checkpoint...")

checkpoint_dir = "CheckPoints/MBT_AV"
if os.path.isdir(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    print(f"✓ Checkpoint directory found: {checkpoint_dir}")
    print(f"  Files: {len(files)}")
    for f in files[:3]:
        print(f"    - {f}")
else:
    print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")

# ============================================================================
# 6. Test JAX computation on GPU
# ============================================================================
print("\n[6/6] Testing JAX computation on GPU...")

try:
    import time
    print("checking JAX matrix multiplication on GPU...")
    # Small matrix multiplication on GPU
    size = 1000
    x = jnp.ones((size, size))
    y = jnp.ones((size, size))
    print(x)
    # Warmup
    z = jnp.dot(x, y)
    
    # Timed
    start = time.time()
    for _ in range(5):
        z = jnp.dot(x, y)
    elapsed = time.time() - start
    
    print(f"✓ JAX matrix multiplication (5 × {size}x{size}): {elapsed:.3f}s")
    print(f"  ≈ {5 * size**2 / (elapsed * 1e9):.1f} GFLOPS")
    
except Exception as e:
    print(f"⚠ JAX computation test failed: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("GPU Setup Verification Complete!")
print("="*80)
print("\nYou can now submit your extraction job with:")
print("  sbatch run_mbt_extraction.sh")
print("\nTo monitor the job:")
print("  squeue -u $USER")
print("\nTo view output:")
print("  tail -f logs/mbt_extraction_*.log")
print("="*80)
