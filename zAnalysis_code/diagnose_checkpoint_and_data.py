#!/usr/bin/env python3
"""Diagnose checkpoint and TFRecord compatibility.

This script:
1. Inspects the checkpoint to infer expected input shapes
2. Reads TFRecords to check actual data shapes
3. Compares them to identify mismatches
4. Infers the correct preprocessing parameters

Usage:
  python diagnose_checkpoint_and_data.py \
    --checkpoint_dir=CheckPoints/MBT_AV \
    --tfrecord_path=Datasets/audioset_eval/data-00000-of-00100.tfrecord \
    --config=scenic/projects/mbt/configs/audioset/Inference_config.py
"""

import os
import pickle
from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from flax.training import checkpoints
import ml_collections

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', None, 'Path to checkpoint directory')
flags.DEFINE_string('tfrecord_path', None, 'Path to a TFRecord file to inspect')
flags.DEFINE_string('config', None, 'Path to config file (optional)')
flags.DEFINE_integer('num_samples', 5, 'Number of TFRecord samples to inspect')

flags.mark_flag_as_required('checkpoint_dir')
flags.mark_flag_as_required('tfrecord_path')


def load_config(config_path: str) -> ml_collections.ConfigDict:
    """Load config from Python file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def inspect_checkpoint(checkpoint_dir: str):
    """Inspect checkpoint structure to infer expected inputs."""
    logging.info(f'\n{"="*80}')
    logging.info('CHECKPOINT INSPECTION')
    logging.info(f'{"="*80}\n')
    
    # Load checkpoint
    if os.path.isfile(checkpoint_dir):
        checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
    else:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                           if not f.startswith('.') and os.path.isfile(os.path.join(checkpoint_dir, f))]
        
        if len(checkpoint_files) == 1:
            single_ckpt = os.path.join(checkpoint_dir, checkpoint_files[0])
            checkpoint_path = checkpoints.restore_checkpoint(single_ckpt, None)
        else:
            checkpoint_path = checkpoints.restore_checkpoint(checkpoint_dir, None)
    
    if checkpoint_path is None:
        raise ValueError(f'No checkpoint found in {checkpoint_dir}')
    
    # Extract params
    if 'params' in checkpoint_path:
        params = checkpoint_path['params']
    elif 'optimizer' in checkpoint_path and 'target' in checkpoint_path['optimizer']:
        params = checkpoint_path['optimizer']['target']
    else:
        params = checkpoint_path
    
    logging.info('Checkpoint structure:')
    
    # Function to recursively print parameter shapes
    def print_params(d, prefix='', max_depth=3, current_depth=0):
        if current_depth > max_depth:
            return
        
        if isinstance(d, dict):
            for key, value in d.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                if hasattr(value, 'shape'):
                    logging.info(f"  {new_prefix}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print_params(value, new_prefix, max_depth, current_depth + 1)
    
    print_params(params, max_depth=2)
    
    # Look for key parameters that indicate input requirements
    logging.info('\n--- Key Parameters ---')
    
    # RGB embedding
    if 'Embedding_rgb' in params:
        emb = params['Embedding_rgb']
        if 'kernel' in emb:
            kernel_shape = emb['kernel'].shape
            logging.info(f'RGB Embedding kernel: {kernel_shape}')
            logging.info(f'  → Patch size: {kernel_shape[0]}x{kernel_shape[1]}')
            logging.info(f'  → Temporal dimension: {kernel_shape[2]}')
            logging.info(f'  → Expected channels: {kernel_shape[3]}')
    
    # Spectrogram embedding
    if 'Embedding_spectrogram' in params:
        emb = params['Embedding_spectrogram']
        if 'kernel' in emb:
            kernel_shape = emb['kernel'].shape
            logging.info(f'Spectrogram Embedding kernel: {kernel_shape}')
            logging.info(f'  → Patch size: {kernel_shape[0]}x{kernel_shape[1]}')
            logging.info(f'  → Expected channels: {kernel_shape[2]}')
    
    # Positional embeddings
    if 'Encoder_rgb' in params or 'Transformer_rgb' in params:
        encoder_key = 'Encoder_rgb' if 'Encoder_rgb' in params else 'Transformer_rgb'
        if 'posembed_input' in params[encoder_key]:
            pos_emb = params[encoder_key]['posembed_input']
            if 'pos_embedding' in pos_emb:
                pos_shape = pos_emb['pos_embedding'].shape
                logging.info(f'RGB Positional Embedding: {pos_shape}')
                num_patches = pos_shape[1] - 1  # Subtract CLS token
                logging.info(f'  → Number of spatial patches: {num_patches}')
                logging.info(f'  → Spatial patches per frame: {int(np.sqrt(num_patches / 4))**2} (assuming 4 temporal patches)')
    
    if 'Encoder_spectrogram' in params or 'Transformer_spectrogram' in params:
        encoder_key = 'Encoder_spectrogram' if 'Encoder_spectrogram' in params else 'Transformer_spectrogram'
        if 'posembed_input' in params[encoder_key]:
            pos_emb = params[encoder_key]['posembed_input']
            if 'pos_embedding' in pos_emb:
                pos_shape = pos_emb['pos_embedding'].shape
                logging.info(f'Spectrogram Positional Embedding: {pos_shape}')
                num_patches = pos_shape[1] - 1  # Subtract CLS token
                logging.info(f'  → Number of spectrogram patches: {num_patches}')
    
    # Bottlenecks
    if 'Bottleneck' in params:
        bn = params['Bottleneck']
        if 'bottleneck' in bn:
            bn_shape = bn['bottleneck'].shape
            logging.info(f'Bottleneck shape: {bn_shape}')
            logging.info(f'  → Number of bottlenecks: {bn_shape[1]}')
    
    # Output projection
    if 'output_projection' in params:
        out = params['output_projection']
        if 'kernel' in out:
            out_shape = out['kernel'].shape
            logging.info(f'Output projection: {out_shape}')
            logging.info(f'  → Number of classes: {out_shape[1]}')
    
    # Pre-logits (if exists)
    if 'pre_logits' in params:
        pre = params['pre_logits']
        if 'kernel' in pre:
            pre_shape = pre['kernel'].shape
            logging.info(f'Pre-logits layer: {pre_shape}')
            logging.info(f'  → Classifier type: GAP (has pre_logits)')
        else:
            logging.info(f'Pre-logits exists but no kernel')
    else:
        logging.info('No pre_logits layer found')
        logging.info('  → Classifier type: TOKEN')
    
    return params


def inspect_tfrecord(tfrecord_path: str, num_samples: int = 5):
    """Inspect TFRecord to see actual data shapes."""
    logging.info(f'\n{"="*80}')
    logging.info('TFRECORD INSPECTION')
    logging.info(f'{"="*80}\n')
    
    if not os.path.exists(tfrecord_path):
        logging.error(f'TFRecord not found: {tfrecord_path}')
        return
    
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    
    for i, raw_record in enumerate(dataset.take(num_samples)):
        logging.info(f'\n--- Sample {i+1} ---')
        
        example = tf.train.SequenceExample()
        example.ParseFromString(raw_record.numpy())
        
        # Check context features (metadata)
        logging.info('Context features:')
        for key in example.context.feature:
            feature = example.context.feature[key]
            if feature.HasField('int64_list'):
                values = feature.int64_list.value
                logging.info(f'  {key}: int64 list, len={len(values)}, values={list(values)[:5]}...')
            elif feature.HasField('float_list'):
                values = feature.float_list.value
                logging.info(f'  {key}: float list, len={len(values)}')
                if key == 'clip/label/multi_hot':
                    label_array = np.array(values)
                    active = np.where(label_array > 0)[0]
                    logging.info(f'    → Active classes: {len(active)} out of {len(values)}')
                    logging.info(f'    → Active indices: {active[:10]}...')
            elif feature.HasField('bytes_list'):
                values = feature.bytes_list.value
                logging.info(f'  {key}: bytes list, len={len(values)}')
        
        # Check sequence features (video/audio frames)
        logging.info('\nSequence features:')
        for key in example.feature_lists.feature_list:
            feature_list = example.feature_lists.feature_list[key]
            num_frames = len(feature_list.feature)
            logging.info(f'  {key}: {num_frames} frames')
            
            if num_frames > 0:
                # Inspect first frame
                first_frame = feature_list.feature[0]
                if first_frame.HasField('bytes_list'):
                    # Decode image/spectrogram to get shape
                    if 'image' in key:
                        try:
                            img = tf.io.decode_jpeg(first_frame.bytes_list.value[0])
                            logging.info(f'    → Frame shape: {img.shape}')
                        except:
                            logging.info(f'    → Could not decode frame')
                    elif 'audio' in key or 'WAVEFORM' in key:
                        try:
                            # Audio might be serialized array
                            audio_bytes = first_frame.bytes_list.value[0]
                            logging.info(f'    → Frame size: {len(audio_bytes)} bytes')
                        except:
                            logging.info(f'    → Could not inspect audio')
                elif first_frame.HasField('float_list'):
                    values = first_frame.float_list.value
                    logging.info(f'    → Frame values: {len(values)} floats')
        
        # Infer preprocessing parameters
        if i == 0:  # Only compute for first sample
            logging.info('\n--- Inferred Preprocessing ---')
            
            # RGB frames
            if 'image/encoded' in example.feature_lists.feature_list:
                num_rgb_frames = len(example.feature_lists.feature_list['image/encoded'].feature)
                logging.info(f'Number of RGB frames: {num_rgb_frames}')
                
                # Try to decode first frame to get resolution
                try:
                    first_frame = example.feature_lists.feature_list['image/encoded'].feature[0]
                    img = tf.io.decode_jpeg(first_frame.bytes_list.value[0])
                    logging.info(f'RGB frame resolution: {img.shape[0]}x{img.shape[1]}')
                    
                    # Infer stride (assuming 25fps and knowing num_frames)
                    # If we have 32 frames, what stride gives us these frames from an 8-second clip?
                    # 8 seconds @ 25fps = 200 total frames
                    # stride = 200 / 32 = 6.25
                    if num_rgb_frames == 32:
                        logging.info('\nAssuming 32 frames extracted from video:')
                        for clip_dur in [8.0, 10.0, 3.0]:
                            total_frames_available = clip_dur * 25
                            stride = total_frames_available / num_rgb_frames
                            logging.info(f'  If clip={clip_dur}s @ 25fps: stride={stride:.2f}')
                except Exception as e:
                    logging.info(f'Could not decode RGB frame: {e}')
            
            # Audio/Spectrogram
            if 'audio' in example.feature_lists.feature_list:
                num_audio_frames = len(example.feature_lists.feature_list['audio'].feature)
                logging.info(f'\nNumber of audio frames: {num_audio_frames}')
                
                # For spectrograms with 100 frames per chunk and hop=10ms:
                # 100 frames × 10ms = 1 second per chunk
                # 8 chunks = 8 seconds
                if num_audio_frames == 8:
                    logging.info('  → Likely 8 seconds of audio (8 chunks × 1s each)')
                    logging.info('  → Each chunk: 100 time steps × 128 mel bins')


def compare_checkpoint_and_data(checkpoint_params, config):
    """Compare checkpoint expectations with config settings."""
    logging.info(f'\n{"="*80}')
    logging.info('CHECKPOINT vs CONFIG COMPARISON')
    logging.info(f'{"="*80}\n')
    
    issues = []
    
    # Compare num_frames
    if config:
        config_num_frames = config.dataset_configs.get('num_frames', None)
        logging.info(f'Config num_frames: {config_num_frames}')
        
        # Try to infer from checkpoint positional embeddings
        # RGB positional embeddings tell us: num_spatial_patches × num_temporal_patches + 1 (CLS)
        # For 32 frames with stride 2 and 3D conv with kernel_size=2:
        # Temporal patches = 32 / 2 = 16... but then 3D conv with kernel=2 reduces to 8? No...
        # Actually with patches.size = [16, 16, 2]:
        # - Spatial: 224/16 = 14 patches per dimension → 14×14 = 196 spatial patches per frame
        # - Temporal: 32 frames, stride 2 in 3D conv... need to think about this
        
        config_stride = config.dataset_configs.get('stride', None)
        logging.info(f'Config stride: {config_stride}')
        
        config_num_spec_frames = config.dataset_configs.get('num_spec_frames', None)
        logging.info(f'Config num_spec_frames: {config_num_spec_frames}')
        
        config_spec_shape = config.dataset_configs.get('spec_shape', None)
        logging.info(f'Config spec_shape: {config_spec_shape}')
        
        config_num_classes = config.dataset_configs.get('num_classes', None)
        logging.info(f'Config num_classes: {config_num_classes}')
        
        # Check output projection
        if 'output_projection' in checkpoint_params and 'kernel' in checkpoint_params['output_projection']:
            ckpt_num_classes = checkpoint_params['output_projection']['kernel'].shape[1]
            logging.info(f'Checkpoint num_classes: {ckpt_num_classes}')
            
            if config_num_classes != ckpt_num_classes:
                issues.append(f'❌ Class mismatch: config has {config_num_classes}, checkpoint has {ckpt_num_classes}')
            else:
                logging.info('✅ Number of classes matches')
        
        # Check bottlenecks
        config_n_bottlenecks = config.model.get('n_bottlenecks', None)
        if 'Bottleneck' in checkpoint_params and 'bottleneck' in checkpoint_params['Bottleneck']:
            ckpt_n_bottlenecks = checkpoint_params['Bottleneck']['bottleneck'].shape[1]
            logging.info(f'\nConfig n_bottlenecks: {config_n_bottlenecks}')
            logging.info(f'Checkpoint bottlenecks: {ckpt_n_bottlenecks}')
            
            if config_n_bottlenecks + 1 == ckpt_n_bottlenecks:
                logging.info('✅ Bottleneck count matches (checkpoint has +1 internal bottleneck)')
            elif config_n_bottlenecks == ckpt_n_bottlenecks:
                logging.info('⚠️  Bottleneck count matches exactly (verify if this is correct)')
            else:
                issues.append(f'❌ Bottleneck mismatch: config has {config_n_bottlenecks}, checkpoint has {ckpt_n_bottlenecks}')
        
        # Check classifier type
        config_classifier = config.model.get('classifier', None)
        has_pre_logits = 'pre_logits' in checkpoint_params
        ckpt_classifier = 'gap' if has_pre_logits else 'token'
        
        logging.info(f'\nConfig classifier: {config_classifier}')
        logging.info(f'Checkpoint classifier: {ckpt_classifier}')
        
        if config_classifier != ckpt_classifier:
            issues.append(f'❌ Classifier mismatch: config uses "{config_classifier}", checkpoint uses "{ckpt_classifier}"')
        else:
            logging.info('✅ Classifier type matches')
    
    if issues:
        logging.info(f'\n{"="*80}')
        logging.info('ISSUES FOUND:')
        logging.info(f'{"="*80}')
        for issue in issues:
            logging.info(issue)
    else:
        logging.info('\n✅ No major issues found between checkpoint and config')
    
    return issues


def main(argv):
    del argv
    
    # Inspect checkpoint
    checkpoint_params = inspect_checkpoint(FLAGS.checkpoint_dir)
    
    # Inspect TFRecords
    inspect_tfrecord(FLAGS.tfrecord_path, FLAGS.num_samples)
    
    # Load and compare with config if provided
    if FLAGS.config:
        logging.info(f'\nLoading config from {FLAGS.config}...')
        config = load_config(FLAGS.config)
        compare_checkpoint_and_data(checkpoint_params, config)
    
    logging.info(f'\n{"="*80}')
    logging.info('DIAGNOSIS COMPLETE')
    logging.info(f'{"="*80}\n')


if __name__ == '__main__':
    app.run(main)
