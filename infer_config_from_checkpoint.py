"""Reverse-engineer training config from checkpoint parameter shapes."""
from flax.training import checkpoints
import numpy as np

def infer_config_from_checkpoint(checkpoint_path, name):
    print(f"\n{'='*70}")
    print(f"Inferring training config for: {name}")
    print(f"{'='*70}")
    
    ckpt = checkpoints.restore_checkpoint(checkpoint_path, None)
    params = ckpt['optimizer']['target']
    
    config = {}
    
    # 1. Classifier type from bottleneck shape
    if 'bottleneck' in params:
        bn_shape = params['bottleneck'].shape
        config['use_bottleneck'] = True
        # If shape[1] = 5 and we assume n_bottlenecks=4, then classifier='token' (adds +1)
        # If shape[1] = 4, then classifier='gap' with n_bottlenecks=4
        if bn_shape[1] == 5:
            config['classifier'] = 'token'
            config['n_bottlenecks'] = 4
        elif bn_shape[1] == 4:
            config['classifier'] = 'gap'
            config['n_bottlenecks'] = 4
        else:
            config['n_bottlenecks'] = bn_shape[1]
            config['classifier'] = 'unknown'
        config['hidden_size'] = bn_shape[2]
    else:
        config['use_bottleneck'] = False
        # Check for CLS tokens to infer classifier
        config['classifier'] = 'token' if 'cls' in params else 'gap'
    
    # 2. Patch size from embedding kernel
    if 'embedding' in params and 'kernel' in params['embedding']:
        emb_shape = params['embedding']['kernel'].shape
        # Shape: (temporal, height, width, channels, hidden)
        config['patches.size'] = [emb_shape[1], emb_shape[2], emb_shape[0]]
        config['hidden_size'] = emb_shape[4]
    
    # 3. Number of layers by counting encoder blocks
    num_layers = 0
    for key in params.keys():
        if key.startswith('encoderblock_') and 'Transformer' in str(params.get(key, {})):
            layer_num = int(key.split('_')[1])
            num_layers = max(num_layers, layer_num + 1)
    
    # Alternative: count in Transformer
    if 'Transformer' in params:
        for key in params['Transformer'].keys():
            if key.startswith('encoderblock_'):
                layer_num = int(key.split('_')[1])
                num_layers = max(num_layers, layer_num + 1)
    
    config['num_layers'] = num_layers
    
    # 4. Number of heads and MLP dim from first encoder block
    transformer = params.get('Transformer', params)
    if 'encoderblock_0' in transformer:
        enc0 = transformer['encoderblock_0']
        
        # Get num_heads from attention layer
        if 'MultiHeadDotProductAttention_0' in enc0:
            attn = enc0['MultiHeadDotProductAttention_0']
            if 'query' in attn and 'kernel' in attn['query']:
                # Query kernel shape: (hidden_size, num_heads, head_dim)
                q_shape = attn['query']['kernel'].shape
                config['num_heads'] = q_shape[1]
                head_dim = q_shape[2]
                print(f"  Attention: query kernel shape = {q_shape}")
                print(f"  → num_heads = {config['num_heads']}, head_dim = {head_dim}")
        
        # Get MLP dim from MlpBlock
        if 'MlpBlock_0' in enc0:
            mlp = enc0['MlpBlock_0']
            if 'Dense_0' in mlp and 'kernel' in mlp['Dense_0']:
                # First dense: (hidden_size, mlp_dim)
                mlp_kernel_shape = mlp['Dense_0']['kernel'].shape
                config['mlp_dim'] = mlp_kernel_shape[1]
                print(f"  MLP: Dense_0 kernel shape = {mlp_kernel_shape}")
                print(f"  → mlp_dim = {config['mlp_dim']}")
    
    # 5. Modality fusion from presence of spectrogram embedding
    modalities = []
    if 'embedding' in params:
        modalities.append('rgb')
    if 'embeddingspectrogram' in params or 'embedding_spectrogram' in params:
        modalities.append('spectrogram')
    if 'clsspectrogram' in params:
        modalities.append('spectrogram')  # Ensure it's there
    config['modality_fusion'] = tuple(set(modalities))
    
    # 6. Fusion layer - try to infer from encoder structure
    # Look for when bottleneck fusion might happen
    config['fusion_layer'] = 'unknown (likely 6 or 8)'
    # Check if we can detect fusion by looking at encoder block structure
    if 'Transformer' in params:
        # Count modality-specific vs fused encoder blocks
        rgb_blocks = 0
        spec_blocks = 0
        fused_blocks = 0
        for key in params['Transformer'].keys():
            if 'encoderblock_' in key:
                # This is tricky - we'd need to check attention patterns
                pass
        # Can't reliably determine fusion_layer from checkpoint alone
    
    # 7. Test settings (can't determine from checkpoint, but critical!)
    config['test_with_bottlenecks'] = 'unknown (should match training)'
    config['temporal_encoding_method'] = 'unknown (likely 3d_conv)'
    config['dropout_rate'] = 'unknown (0.0 for inference, but what was training?)'
    
    # 8. Output projection (classifier head)
    if 'output_projection' in params:
        op = params['output_projection']
        if 'kernel' in op:
            op_shape = op['kernel'].shape
            config['num_classes'] = op_shape[1]
            print(f"  Output projection kernel shape = {op_shape}")
            print(f"  → num_classes = {config['num_classes']}")
        if 'bias' in op:
            bias = np.array(op['bias'])
            config['output_bias_mean'] = float(bias.mean())
            config['output_bias_std'] = float(bias.std())
    
    # 9. Representation size
    config['representation_size'] = None  # Not used in typical configs
    if 'pre_logits' in params:
        config['representation_size'] = 'present (size unknown from params alone)'
    
    # 10. CRITICAL: Check for batch normalization or layer norm stats
    # These would be in model_state, not params
    print(f"\n  Checking for normalization stats...")
    if 'model_state' in ckpt and ckpt['model_state']:
        print(f"  ⚠️  model_state is NOT empty - contains batch_stats or other state!")
        config['has_batch_stats'] = True
        # This could explain poor performance if we're not loading them!
    else:
        print(f"  ✓ model_state is empty (no batch_stats)")
        config['has_batch_stats'] = False
    
    return config

# Check both checkpoints
for name, path in [
    ('MBT_AV', 'CheckPoints/MBT_AV/mbtb32_as-500k_rgb-spec'),
    ('MINI_AV', 'CheckPoints/MINI_AV/mbtb32_as-mini_rgb-spec')
]:
    config = infer_config_from_checkpoint(path, name)
    
    print(f"\n{'='*70}")
    print(f"INFERRED CONFIG for {name}:")
    print(f"{'='*70}")
    for key, value in sorted(config.items()):
        print(f"  config.model.{key} = {value}")
    print()

print(f"{'='*70}")
print("POTENTIAL MISMATCHES:")
print("These settings can't be determined from checkpoint but affect results:")
print("  1. fusion_layer (6 vs 8 makes a big difference!)")
print("  2. test_with_bottlenecks (True vs False)")
print("  3. temporal_encoding_config.method ('3d_conv' vs others)")
print("  4. Training hyperparameters (learning rate, augmentation, etc.)")
print("  5. Dataset preprocessing (spec_mean, spec_stddev, normalization)")
print("\nThe checkpoints ARE trained with classifier='token', but performance")
print("could still be poor due to these other mismatches or undertraining.")
print(f"{'='*70}")
