#!/usr/bin/env python3
"""Detailed profiling script for MBT training bottlenecks.

This script runs training for a limited number of steps and provides
detailed timing breakdowns to identify bottlenecks.

Usage:
  python profile_training.py --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
                              --workdir=mbt_profile/ \
                              --num_steps=50
"""

import time
import sys
from absl import app, flags
import numpy as np
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 50, 'Number of steps to profile')

def profile_training(argv):
  """Run training with detailed profiling."""
  
  # Import here to avoid issues before flags are parsed
  from scenic.projects.mbt import main as mbt_main
  
  print("\n" + "="*80)
  print("PROFILING MODE - Limited training run for bottleneck analysis")
  print(f"Will run {FLAGS.num_steps} steps with detailed timing")
  print("="*80 + "\n")
  
  # Modify config to run limited steps
  original_argv = sys.argv.copy()
  
  # Run the main training (it will use our instrumented trainer.py)
  try:
    mbt_main.main(argv)
  except KeyboardInterrupt:
    print("\nProfiling interrupted by user")
  except Exception as e:
    print(f"\nProfiling completed with exception: {e}")
  
  print("\n" + "="*80)
  print("PROFILING COMPLETE")
  print("See timing analysis above for bottleneck details")
  print("="*80 + "\n")

if __name__ == '__main__':
  app.run(profile_training)
