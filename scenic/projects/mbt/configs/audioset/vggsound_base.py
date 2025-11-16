# Copyright 2025 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Multimodal sound classification on VGGSound dataset.

VGGSound dataset configuration for MBT model training.
This config uses TFRecords preprocessed from VGGSound videos.
"""
# pylint: disable=line-too-long

import glob
import ml_collections
import os

# VGGSound dataset sizes (update based on your actual preprocessed data)
VGGSOUND_TRAIN_SIZE = 183971  # Actual count from your train CSV
VGGSOUND_TEST_SIZE = 15496    # Actual count from your test CSV


def get_config():
  """Returns the VGGSound experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'mbt_vggsound_classification'
  config.dataset_name = 'audiovisual_tfrecord_dataset'

  # Dataset - UPDATE THESE PATHS TO YOUR EXTRACTED TFRECORDS
  config.dataset_configs = ml_collections.ConfigDict()
  
  # Point base_dir to the parent directory containing train_tfrecords_local and test_tfrecords_local
  config.dataset_configs.base_dir = '/home/labuta/Documents/Bram/scenic_PhD'
  
  # Expand glob patterns to get actual file lists
  train_pattern = os.path.join(config.dataset_configs.base_dir, 'train_tfrecords_local/tar*_batch*/data-*-of-*.tfrecord')
  test_pattern = os.path.join(config.dataset_configs.base_dir, 'test_tfrecords_local/tar*_batch*/data-*-of-*.tfrecord')
  
  train_files = sorted(glob.glob(train_pattern))
  test_files = sorted(glob.glob(test_pattern))
  
  # Convert to relative paths from base_dir
  train_files = [os.path.relpath(f, config.dataset_configs.base_dir) for f in train_files]
  test_files = [os.path.relpath(f, config.dataset_configs.base_dir) for f in test_files]
  
  config.dataset_configs.tables = {
      'train': train_files,
      'validation': test_files,
      'test': test_files,
  }
  
  config.dataset_configs.examples_per_subset = {
      'train': VGGSOUND_TRAIN_SIZE,
      'validation': VGGSOUND_TEST_SIZE,
      'test': VGGSOUND_TEST_SIZE
  }
  
  # VGGSound has 309 classes (vs AudioSet's 527)
  config.dataset_configs.num_classes = 309
  config.data_dtype_str = 'float32'
  
  # List of modalities to load, supports `rgb` and `spectrogram'.
  config.dataset_configs.modalities = ('spectrogram', 'rgb')
  # Must be True for multi-modal training (when using both RGB and spectrogram)
  # Model returns dict during training, pooled output during eval
  config.dataset_configs.return_as_dict = True
  
  # VGGSound videos are 10 seconds at 25fps = 250 frames
  config.dataset_configs.num_frames = 32  # Paper spec: 8 frames
  config.dataset_configs.stride = 2  # Uniform stride over sampling window

  config.dataset_configs.num_spec_frames = 8  # chunkes instead of 800 individuals
  config.dataset_configs.spec_stride = 1

  # Audio spectrogram statistics (you may need to calculate these from your data)
  # For now, using AudioSet values - consider computing VGGSound-specific stats
  config.dataset_configs.spec_mean = 1.102
  config.dataset_configs.spec_stddev = 2.762

  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  # VGGSound TFRecords: each frame in the sequence has shape (1, 128)
  # We'll sample num_spec_frames=100 frames to get a (100, 128) spectrogram
  config.dataset_configs.spec_shape = (100, 128)

  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True
  config.dataset_configs.log_test_epochs = 4
  config.dataset_configs.num_test_clips = 4
  config.dataset_configs.test_batch_size = 1  # Must equal number of GPUs (jax.local_device_count())
  config.multicrop_clips_per_device = 2

  # Data augmentation
  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = True
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  # Aggressive prefetching to keep GPU fed with data (SSD storage)
  config.dataset_configs.prefetch_to_device = 16  # Increased from 8 - allows more batches queued on GPU
  config.dataset_configs.prefetch_to_host = 8     # Increased from 4 - more CPU RAM buffering
  
  # TFRecord reading parallelism (for SSD storage)
  config.dataset_configs.num_parallel_calls = 8  # Parallel decompression/decoding threads
  config.dataset_configs.cycle_length = 8        # Number of TFRecord files to read in parallel

  # SpecAugment hyperparameters - Paper spec: max time=192, max freq=48
  # Reduced time_mask_count for speed (paper uses 4, we use 2 for 2x faster augmentation)
  config.dataset_configs.spec_augment = True
  config.dataset_configs.spec_augment_params = ml_collections.ConfigDict()
  config.dataset_configs.spec_augment_params.freq_mask_max_bins = 48  # Paper spec
  config.dataset_configs.spec_augment_params.freq_mask_count = 1
  config.dataset_configs.spec_augment_params.time_mask_max_frames = 48  # Paper spec (was 48)
  config.dataset_configs.spec_augment_params.time_mask_count = 4  # Reduced from 4 for speed (still effective regularization)
  config.dataset_configs.spec_augment_params.time_warp_max_frames = 1.0
  config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
  config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0

  # Model: MBT-base for single-label multi-class classification
  config.model_name = 'mbt_classification'  # Use softmax, not sigmoid
  config.model = ml_collections.ConfigDict()
  config.model.modality_fusion = ('spectrogram', 'rgb')
  config.model.use_bottleneck = True
  config.model.test_with_bottlenecks = True
  config.model.share_encoder = False
  config.model.n_bottlenecks = 4
  config.model.fusion_layer = 8
  config.model.hidden_size = 768
  config.model.patches = ml_collections.ConfigDict()
  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'spacetime'
  config.model.num_heads = 12
  config.model.mlp_dim = 3072
  config.model.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'gap'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = [16, 16, 2]
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  config.model.temporal_encoding_config.n_sampled_frames = 4

  # Training
  config.trainer_name = 'mbt_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = 0.3
  config.num_training_epochs = 50
  config.batch_size = 8  # Reduced from 12 due to larger spectrograms (800 vs 100 frames)
  config.rng_seed = 0
  
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.3  # Paper spec (was 0.5)
  config.mixmod = False
  config.model.stochastic_droplayer_rate = 0.3

  # Pre-trained weights initialization
  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_config = None 
  config.init_from.checkpoint_path = "/home/labuta/Documents/Bram/scenic_PhD/CheckPoints/ViT_B_16_ImageNet1k_dir"  # Set to pretrained ViT path if available
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.model_config = ml_collections.ConfigDict()
  config.init_from.model_config.model = ml_collections.ConfigDict()
  config.init_from.model_config.model.classifier = 'token'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'resize_tile'

  # Learning rate schedule
  steps_per_epoch = VGGSOUND_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 2.5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 5e-1

  # Logging
  config.write_summary = True
  config.checkpoint = False  # Temporarily disable to debug step 2 issue
  config.debug_train = False
  config.debug_eval = False
  config.checkpoint_steps = 500
  
  # Eval every 5 epochs instead of every epoch to speed up training
  # This reduces eval overhead from 50% to ~10%
  config.log_eval_steps = 5 * steps_per_epoch  # Eval every 5 epochs
  
  return config
