#!/usr/bin/env python3
"""Infer the training configuration from checkpoint structure.

Key parameters we can detect:
1. classifier type ('gap' vs 'token') - from presence of 'pre_logits' layer
2. number of bottlenecks - from 'bottleneck' parameters
3. fusion layer - from when spectrogram encoder blocks stop
4. number of layers - from total encoder blocks
"""

import numpy as np
from flax.training import checkpoints

checkpoint_path = 'CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec'

print("Loading checkpoint...")
checkpoint = checkpoints.restore_checkpoint(checkpoint_path, None)

params = checkpoint['optimizer']['target']

print("\n" + "="*80)
print("CHECKPOINT STRUCTURE ANALYSIS:")
print("="*80)

# 1. Check classifier type
print("\n1. CLASSIFIER TYPE:")
print("-" * 40)
has_pre_logits = 'pre_logits' in params
has_output_projection = 'output_projection' in params

if has_pre_logits:
    print("✓ Found 'pre_logits' layer → classifier='token'")
    print("  (Token classifier uses CLS token with pre_logits projection)")
else:
    print("✗ No 'pre_logits' layer → classifier='gap'")
    print("  (GAP classifier directly pools encoder outputs)")

if has_output_projection:
    print("✓ Found 'output_projection' layer (final classifier)")
    op_shape = np.array(params['output_projection']['kernel']).shape
    print(f"  Shape: {op_shape} (hidden_dim={op_shape[0]}, num_classes={op_shape[1]})")

# 2. Count bottleneck tokens
print("\n2. BOTTLENECK CONFIGURATION:")
print("-" * 40)
if 'bottleneck' in params:
    bn_shape = np.array(params['bottleneck']).shape
    print(f"✓ Found bottleneck with shape: {bn_shape}")
    print(f"  Number of bottlenecks: {bn_shape[0]}")
    print(f"  Hidden dimension: {bn_shape[1]}")
else:
    print("✗ No bottleneck found → use_bottleneck=False")

# 3. Count encoder blocks
print("\n3. ENCODER ARCHITECTURE:")
print("-" * 40)
rgb_blocks = []
spec_blocks = []

for key in params.get('Transformer', {}).keys():
    if 'encoderblock_' in key:
        if '_spectrogram' in key:
            block_num = int(key.replace('encoderblock_', '').replace('_spectrogram', ''))
            spec_blocks.append(block_num)
        else:
            block_num = int(key.replace('encoderblock_', ''))
            rgb_blocks.append(block_num)

rgb_blocks = sorted(set(rgb_blocks))
spec_blocks = sorted(set(spec_blocks))

print(f"RGB encoder blocks: {rgb_blocks}")
print(f"Spectrogram encoder blocks: {spec_blocks}")
print(f"Total RGB layers: {len(rgb_blocks)}")
print(f"Total spectrogram layers: {len(spec_blocks)}")

if len(spec_blocks) < len(rgb_blocks):
    fusion_layer = len(spec_blocks)
    print(f"\n→ Fusion layer: {fusion_layer}")
    print(f"  (Spectrogram encoder stops at layer {fusion_layer}, then fuses with RGB)")
else:
    print("\n→ No clear fusion point detected")

# 4. Check model dimensions
print("\n4. MODEL DIMENSIONS:")
print("-" * 40)
if 'Transformer' in params and rgb_blocks:
    first_block = f'encoderblock_{rgb_blocks[0]}'
    if first_block in params['Transformer']:
        mlp_key = f'Transformer/{first_block}/MlpBlock_0/Dense_0/kernel'
        
        # Navigate to the kernel
        block_params = params['Transformer'][first_block]
        if 'MlpBlock_0' in block_params:
            mlp_params = block_params['MlpBlock_0']
            if 'Dense_0' in mlp_params:
                dense_params = mlp_params['Dense_0']
                if 'kernel' in dense_params:
                    kernel = np.array(dense_params['kernel'])
                    hidden_size = kernel.shape[0]
                    mlp_dim = kernel.shape[1]
                    print(f"Hidden size: {hidden_size}")
                    print(f"MLP dimension: {mlp_dim}")

# 5. Check attention heads
print("\n5. ATTENTION CONFIGURATION:")
print("-" * 40)
if 'Transformer' in params and rgb_blocks:
    first_block = f'encoderblock_{rgb_blocks[0]}'
    if first_block in params['Transformer']:
        attn_params = params['Transformer'][first_block]['MultiHeadDotProductAttention_0']
        if 'query' in attn_params and 'kernel' in attn_params['query']:
            query_kernel = np.array(attn_params['query']['kernel'])
            print(f"Query kernel shape: {query_kernel.shape}")
            if len(query_kernel.shape) == 3:
                hidden_dim, num_heads, head_dim = query_kernel.shape
                print(f"  Hidden dimension: {hidden_dim}")
                print(f"  Number of heads: {num_heads}")
                print(f"  Head dimension: {head_dim}")

# 6. Check CLS tokens
print("\n6. CLS TOKEN CONFIGURATION:")
print("-" * 40)
if 'cls' in params:
    cls_shape = np.array(params['cls']).shape
    print(f"✓ Found 'cls' token with shape: {cls_shape}")
    
if 'clsspectrogram' in params:
    cls_spec_shape = np.array(params['clsspectrogram']).shape
    print(f"✓ Found 'clsspectrogram' token with shape: {cls_spec_shape}")

print("\n" + "="*80)
print("INFERRED CONFIGURATION:")
print("="*80)

config_dict = {
    'model.classifier': 'token' if has_pre_logits else 'gap',
    'model.num_layers': len(rgb_blocks),
    'model.fusion_layer': len(spec_blocks) if len(spec_blocks) < len(rgb_blocks) else 'unknown',
    'model.use_bottleneck': 'bottleneck' in params,
}

if 'bottleneck' in params:
    config_dict['model.n_bottlenecks'] = np.array(params['bottleneck']).shape[0]

for key, value in config_dict.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("COMPARISON WITH AVAILABLE CONFIGS:")
print("="*80)

# balanced_audioset_base.py
print("\nbalanced_audioset_base.py:")
print("  classifier: 'gap'")
print("  num_layers: 12")
print("  fusion_layer: 8")
print("  n_bottlenecks: 4")
print("  hidden_size: 768")
print("  num_heads: 12")

# Inference_config.py
print("\nInference_config.py:")
print("  classifier: 'token'")
print("  num_layers: 12")
print("  fusion_layer: 8")
print("  n_bottlenecks: 4")
print("  hidden_size: 768")
print("  num_heads: 12")

print("\n" + "="*80)
if has_pre_logits:
    print("RESULT: Checkpoint was trained with classifier='token'")
    print("→ Use Inference_config.py (or set classifier='token')")
else:
    print("RESULT: Checkpoint was trained with classifier='gap'")
    print("→ Use balanced_audioset_base.py (or set classifier='gap')")
print("="*80)
