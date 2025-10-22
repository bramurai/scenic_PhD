
"""Example script for extracting MBT model activations.

This demonstrates how to:
1. Load a pretrained MBT model
2. Extract activations from different layers
3. Analyze activation patterns
4. Save results for further analysis
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags, logging
import ml_collections
from pathlib import Path

from scenic.projects.mbt import inference
from scenic.projects.mbt.configs.audioset import balanced_audioset_base


FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_path', None, 'Path to model checkpoint')
flags.DEFINE_string('output_dir', 'activation_analysis', 'Directory to save results')
flags.DEFINE_string('dataset_path', None, 'Path to dataset directory (overrides config)')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference')
flags.DEFINE_bool('use_float16', False, 'Use float16 for memory efficiency')
flags.DEFINE_enum('input_source', 'dummy', ['dummy', 'dataset', 'file'],
                  'Source of input data: dummy (zeros), dataset (load from tfrecords), or file (numpy)')
flags.DEFINE_string('rgb_file', None, 'Path to numpy file with RGB data (if input_source=file)')
flags.DEFINE_string('spec_file', None, 'Path to numpy file with spectrogram data (if input_source=file)')


def modify_config_for_activation_extraction(config: ml_collections.ConfigDict):
    """Modify config to enable activation extraction."""
    
    # Set batch size
    config.batch_size = FLAGS.batch_size
    
    # Disable all stochastic operations
    config.model.dropout_rate = 0.0
    config.model.attention_dropout_rate = 0.0
    config.model.stochastic_droplayer_rate = 0.0
    
    # Disable data augmentation
    config.dataset_configs.spec_augment = False
    if hasattr(config.dataset_configs, 'augmentation_params'):
        config.dataset_configs.augmentation_params.do_color_augment = False
        config.dataset_configs.augmentation_params.do_jitter_scale = False
        config.dataset_configs.augmentation_params.prob_scale_jitter = 0.0
    
    # IMPORTANT: Ensure classifier matches checkpoint
    # Based on checkpoint inspection, the pretrained model used:
    # - classifier='token' (not 'gap')
    # - This adds CLS tokens and creates n_bottlenecks + 1 total bottleneck tokens
    if FLAGS.checkpoint_path:
        logging.info("Checkpoint provided - adjusting config to match checkpoint")
        config.model.classifier = 'token'
        config.model.n_bottlenecks = 4  # With classifier='token', this becomes 5 total
        logging.info("  Using classifier='token' with n_bottlenecks=4 (5 total including CLS)")
    
    # Optional: use float16 for memory efficiency
    if FLAGS.use_float16:
        config.model_dtype_str = 'float16'
        logging.info("Using float16 precision")
    
    # Enable intermediate output extraction
    # Option 1: Get all token embeddings
    config.model.return_preclassifier = False
    
    # Option 2: Get pre-classification features
    config.model.return_prelogits = False
    
    return config


def extract_patch_embeddings(model, variables, input_data):
    """Extract patch embeddings from the input.
    
    This captures the output after temporal_encode but before the transformer.
    """
    # You would need to modify the model to return these
    # For now, we can only get final outputs
    logging.warning("Patch embedding extraction requires model modification")
    return None


def extract_modality_specific_features(results, config):
    """Extract features specific to each modality (RGB vs Spectrogram).
    
    Args:
        results: Inference results
        config: Model config
        
    Returns:
        Dictionary with modality-specific features
    """
    modality_features = {}
    
    # Check if model returns dict (modality-specific outputs)
    if isinstance(results['outputs'], dict):
        for modality in results['outputs']:
            modality_features[modality] = results['outputs'][modality]
            logging.info(f"Extracted features for {modality}: {modality_features[modality].shape}")
    else:
        logging.info(f"Model returns fused output: {results['outputs'].shape}")
        modality_features['fused'] = results['outputs']
    
    return modality_features


def extract_bottleneck_tokens(config, checkpoint_path, input_data):
    """Extract bottleneck tokens if model uses bottleneck fusion.
    
    The bottleneck tokens are learned tokens that mediate cross-modal interaction.
    """
    if not config.model.use_bottleneck:
        logging.info("Model does not use bottleneck fusion")
        return None
    
    logging.info(f"Model uses {config.model.n_bottlenecks} bottleneck tokens")
    logging.info(f"Fusion happens at layer {config.model.fusion_layer}")
    
    # To extract bottleneck tokens, we'd need to modify the Encoder
    # to return them explicitly
    logging.warning("Bottleneck token extraction requires model modification")
    return None


def analyze_attention_patterns(config):
    """Analyze cross-modal attention patterns.
    
    This requires modifying MultiHeadDotProductAttention to return attention weights.
    """
    logging.info(f"Model has {config.model.num_layers} transformer layers")
    logging.info(f"Each layer has {config.model.num_heads} attention heads")
    logging.info(f"Cross-modal fusion starts at layer {config.model.fusion_layer}")
    
    logging.warning("Attention pattern extraction requires model modification")
    logging.info("You would need to modify EncoderBlock to return attention weights")
    
    return None


def load_input_data(config, source='dummy'):
    """Load input data from various sources.
    
    Args:
        config: Model configuration
        source: One of 'dummy', 'dataset', or 'file'
        
    Returns:
        Dictionary with 'rgb' and 'spectrogram' tensors
    """
    batch_size = config.batch_size
    
    if source == 'dummy':
        logging.info("Creating dummy (zero) input data...")
        test_input = {
            'rgb': jnp.zeros((batch_size, 32, 224, 224, 3), dtype=jnp.float32),
            'spectrogram': jnp.zeros((batch_size, 800, 128, 3), dtype=jnp.float32)
        }
        logging.info("  ⚠️  Using all-zero data (for architecture analysis only)")
        
    elif source == 'dataset':
        logging.info("Loading real data from dataset...")
        from scenic.train_lib import train_utils
        
        # Override dataset config with your custom dataset path
        if FLAGS.dataset_path:
            logging.info(f"Using custom dataset path: {FLAGS.dataset_path}")
            config.dataset_configs.base_dir = FLAGS.dataset_path
            
            # Update table names to match single-file format
            # Your file is named "test-00000-of-00001"
            config.dataset_configs.tables = {
                'train': 'test-00000-of-00001',
                'validation': 'test-00000-of-00001', 
                'test': 'test-00000-of-00001',
            }
            logging.info(f"Updated table names to: {config.dataset_configs.tables}")
        
        try:
            # Create dataset iterator
            rng = jax.random.PRNGKey(config.get('rng_seed', 0))
            dataset = train_utils.get_dataset(config, rng)
            
            # Get first batch from validation set
            test_batch = next(iter(dataset.valid_iter))
            test_input = {}
            
            # Extract modalities
            if 'rgb' in test_batch['inputs']:
                test_input['rgb'] = test_batch['inputs']['rgb'][:batch_size]
            if 'spectrogram' in test_batch['inputs']:
                test_input['spectrogram'] = test_batch['inputs']['spectrogram'][:batch_size]
                
            logging.info("  ✓ Loaded real data from TFRecord dataset")
            
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            logging.error("\n" + "="*80)
            logging.error("TROUBLESHOOTING:")
            logging.error("Your TFRecord file format may not match the expected structure.")
            logging.error(f"Expected format: AudioSet with both RGB video and mel-spectrogram")
            logging.error(f"Your file: {FLAGS.dataset_path}/test-00000-of-00001")
            logging.error("\nOptions:")
            logging.error("1. Use --input_source=dummy to test with synthetic data")
            logging.error("2. Use --input_source=file with --rgb_file and --spec_file")
            logging.error("3. Recreate TFRecord with correct format (see AudioSet preprocessing)")
            logging.error("="*80)
            raise
        
    elif source == 'file':
        logging.info("Loading from numpy files...")
        if not FLAGS.rgb_file or not FLAGS.spec_file:
            raise ValueError("Must specify --rgb_file and --spec_file when using input_source=file")
        
        test_input = {
            'rgb': jnp.array(np.load(FLAGS.rgb_file)),
            'spectrogram': jnp.array(np.load(FLAGS.spec_file))
        }
        
        # Take only batch_size samples
        test_input['rgb'] = test_input['rgb'][:batch_size]
        test_input['spectrogram'] = test_input['spectrogram'][:batch_size]
        
        logging.info(f"  ✓ Loaded from files: {FLAGS.rgb_file}, {FLAGS.spec_file}")
    
    else:
        raise ValueError(f"Unknown input source: {source}")
    
    # Log shapes
    logging.info(f"  RGB shape: {test_input['rgb'].shape}")
    logging.info(f"  Spectrogram shape: {test_input['spectrogram'].shape}")
    
    return test_input


def main(_):
    """Main function to extract and analyze activations."""
    
    # Create output directory
    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {output_dir}")
    
    # 1. Load and modify config
    logging.info("Loading configuration...")
    config = balanced_audioset_base.get_config()
    config = modify_config_for_activation_extraction(config)
    
    # 2. Load input data
    logging.info(f"\nLoading input data (source: {FLAGS.input_source})...")
    test_input = load_input_data(config, source=FLAGS.input_source)
    
    # 3. Run inference
    if FLAGS.checkpoint_path:
        logging.info(f"Loading checkpoint from {FLAGS.checkpoint_path}")
        # Start with basic inference (no intermediates) to verify checkpoint works
        results = inference.run_inference(
            config,
            FLAGS.checkpoint_path,
            test_input,
            extract_intermediates=False  # Set to True once basic inference works
        )
    else:
        logging.warning("No checkpoint provided, using random initialization")
        # Still useful for analyzing architecture
        dataset_meta_data = inference.build_dataset_metadata(config)
        from scenic.projects.mbt import model as mbt_model
        model_cls = mbt_model.MBTMultilabelClassificationModel
        model_instance = model_cls(config, dataset_meta_data)
        
        # Initialize model
        rng = jax.random.PRNGKey(0)
        
        # IMPORTANT: Make a copy of test_input for initialization
        # The model modifies the input dict in-place during forward pass
        test_input_init = {k: v for k, v in test_input.items()}
        
        variables = model_instance.flax_model.init(
            rng, test_input_init, train=False, debug=False
        )
        
        # Run forward pass with original test_input
        outputs = model_instance.flax_model.apply(
            variables, test_input, train=False, mutable=False
        )
        results = {'outputs': outputs}
    
    # 4. Extract and analyze features
    logging.info("\n" + "="*80)
    logging.info("ACTIVATION ANALYSIS")
    logging.info("="*80)
    
    # Final output shape
    logging.info(f"\nFinal output shape: {results['outputs'].shape}")
    logging.info(f"Number of classes: {config.dataset_configs.num_classes}")
    
    # Modality-specific features
    modality_features = extract_modality_specific_features(results, config)
    
    # Bottleneck analysis
    bottleneck_info = extract_bottleneck_tokens(config, FLAGS.checkpoint_path, test_input)
    
    # Attention analysis
    attention_info = analyze_attention_patterns(config)
    
    # 5. Analyze activations
    if 'intermediates' in results:
        logging.info("\nAnalyzing intermediate activations...")
        analysis = inference.analyze_activations(
            results['intermediates'],
            save_path=str(output_dir / 'activation_stats.npy')
        )
    
    # 6. Save results
    logging.info(f"\nSaving results to {output_dir}")
    
    # Save predictions
    np.save(output_dir / 'predictions.npy', np.array(results['outputs']))
    
    # Save modality features
    for modality, features in modality_features.items():
        np.save(output_dir / f'{modality}_features.npy', np.array(features))
    
    # Save config
    with open(output_dir / 'config.txt', 'w') as f:
        f.write(str(config))
    
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"Model type: {config.model_name}")
    logging.info(f"Modalities: {config.model.modality_fusion}")
    logging.info(f"Hidden size: {config.model.hidden_size}")
    logging.info(f"Number of layers: {config.model.num_layers}")
    logging.info(f"Number of heads: {config.model.num_heads}")
    logging.info(f"Fusion layer: {config.model.fusion_layer}")
    logging.info(f"Use bottleneck: {config.model.use_bottleneck}")
    logging.info(f"Classifier type: {config.model.classifier}")
    logging.info("="*80)
    
    logging.info(f"\n✓ Analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    app.run(main)
