#!/usr/bin/env python3
"""Test both classifier='token' and classifier='gap' to see which works."""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from scenic.projects.mbt import model as mbt_model
import ml_collections

checkpoint_path = 'CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec'

# Load checkpoint
print("Loading checkpoint...")
checkpoint = checkpoints.restore_checkpoint(checkpoint_path, None)
params = checkpoint['optimizer']['target']
model_state = {}

# Create dummy input
batch_size = 1
rgb_input = np.random.randn(batch_size, 32, 224, 224, 3).astype(np.float32)
spec_input = np.random.randn(batch_size, 800, 128, 3).astype(np.float32)

inputs = {
    'rgb': rgb_input,
    'spectrogram': spec_input
}

print(f"\nInput shapes:")
print(f"  RGB: {rgb_input.shape}")
print(f"  Spectrogram: {spec_input.shape}")

def test_classifier(classifier_type, n_bottlenecks):
    """Test a specific classifier configuration."""
    print(f"\n{'='*80}")
    print(f"Testing classifier='{classifier_type}' with n_bottlenecks={n_bottlenecks}")
    print(f"{'='*80}")
    
    config = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    config.model.modality_fusion = ('spectrogram', 'rgb')
    config.model.use_bottleneck = True
    config.model.test_with_bottlenecks = True
    config.model.share_encoder = False
    config.model.n_bottlenecks = n_bottlenecks
    config.model.fusion_layer = 8
    config.model.hidden_size = 768
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [16, 16, 2]
    config.model.attention_config = ml_collections.ConfigDict()
    config.model.attention_config.type = 'spacetime'
    config.model.num_heads = 12
    config.model.mlp_dim = 3072
    config.model.num_layers = 12
    config.model.representation_size = None
    config.model.classifier = classifier_type
    config.model.attention_dropout_rate = 0.
    config.model.dropout_rate = 0.
    config.model.temporal_encoding_config = ml_collections.ConfigDict()
    config.model.temporal_encoding_config.method = '3d_conv'
    config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
    config.model.temporal_encoding_config.n_sampled_frames = 4
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.num_spec_frames = 8
    config.dataset_configs.spec_shape = (100, 128)
    config.dataset_configs.num_frames = 32
    
    model_cls = mbt_model.MBTMultilabelClassificationModel
    spec_time_dim = 8 * 100
    
    model_instance = model_cls(config, {
        'num_classes': 527,
        'input_shape': {
            'rgb': (-1, 32, 224, 224, 3),
            'spectrogram': (-1, spec_time_dim, 128, 3)
        },
        'input_dtype': jnp.float32,
        'target_is_onehot': True
    })
    
    try:
        variables = {'params': params}
        if model_state:
            variables['batch_stats'] = model_state
        
        output = model_instance.flax_model.apply(
            variables,
            inputs,
            train=False
        )
        
        output_np = np.array(output)
        print(f"✓ SUCCESS!")
        print(f"  Output shape: {output_np.shape}")
        print(f"  Output stats:")
        print(f"    Min: {np.min(output_np):.4f}")
        print(f"    Max: {np.max(output_np):.4f}")
        print(f"    Mean: {np.mean(output_np):.4f}")
        print(f"    Std: {np.std(output_np):.4f}")
        
        # Apply sigmoid and check
        probs = 1.0 / (1.0 + np.exp(-output_np))
        print(f"  After sigmoid:")
        print(f"    Min prob: {np.min(probs):.4f}")
        print(f"    Max prob: {np.max(probs):.4f}")
        print(f"    Mean prob: {np.mean(probs):.4f}")
        print(f"  Top 5 predictions (class indices): {np.argsort(output_np[0])[-5:][::-1]}")
        
        return True, output_np
        
    except Exception as e:
        print(f"✗ FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# Test both configurations
print("\n" + "="*80)
print("TESTING DIFFERENT CLASSIFIER CONFIGURATIONS")
print("="*80)

results = {}

# Test 1: classifier='token', n_bottlenecks=4 (creates 5 bottlenecks)
success, output = test_classifier('token', 4)
results['token_n4'] = (success, output)

# Test 2: classifier='gap', n_bottlenecks=5
success, output = test_classifier('gap', 5)
results['gap_n5'] = (success, output)

# Test 3: classifier='gap', n_bottlenecks=4
success, output = test_classifier('gap', 4)
results['gap_n4'] = (success, output)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for config_name, (success, output) in results.items():
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{config_name}: {status}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
if results['token_n4'][0]:
    print("Use classifier='token' with n_bottlenecks=4")
    print("(This creates 5 bottleneck tokens total: 4 + 1 for token classifier)")
elif results['gap_n5'][0]:
    print("Use classifier='gap' with n_bottlenecks=5")
else:
    print("Unable to determine - both configurations failed!")
