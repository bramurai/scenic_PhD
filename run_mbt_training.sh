#!/bin/bash

# Disable GPU graph-level fusion which triggers CUDA graph capture
# This must be set as environment variable before JAX/XLA initializes
export XLA_FLAGS="--xla_disable_hlo_passes=gpu-graph-level-fusion"

# Run the MBT training  
cd /home/labuta/Documents/Bram/scenic_PhD
conda run -n scenic_phd --no-capture-output python -m scenic.projects.mbt.main \
    --config=scenic/projects/mbt/configs/audioset/vggsound_base.py \
    --workdir=mbt_base/
