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
r"""Inference configuration for neural activation and attention analysis.

This config is designed to:
1. Load 9 test samples from Audioset_test
2. Extract layer-wise activations
3. Extract attention weights from all attention layers
4. Save outputs for analysis
"""
# pylint: disable=line-too-long

import ml_collections

# 9 test samples in Audioset_test folder
AUDIOSET_TEST_SIZE = 9


def get_config():
  """Returns the inference configuration for activation analysis."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'mbt_activation_analysis'
  config.dataset_name = 'audiovisual_tfrecord_dataset'

  # Dataset - pointing to Audioset_test folder with 9 samples
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.base_dir = 'Audioset_test'
  config.dataset_configs.tables = {
      'train': ['data-00000-of-00001.tfrecord'],  # Use test data as "train" for simplicity
      'validation': ['data-00000-of-00001.tfrecord'],
      'test': ['data-00000-of-00001.tfrecord'],
  }
  config.dataset_configs.examples_per_subset = {
      'train': AUDIOSET_TEST_SIZE,
      'validation': AUDIOSET_TEST_SIZE,
      'test': AUDIOSET_TEST_SIZE
  }
  config.dataset_configs.num_classes = 527
  config.data_dtype_str = 'float32'
  
  # List of modalities to load
  config.dataset_configs.modalities = ('spectrogram', 'rgb')
  config.dataset_configs.return_as_dict = True
  
  # Match the preprocessing from training
  config.dataset_configs.num_frames = 8
  config.dataset_configs.stride = 8
  config.dataset_configs.num_spec_frames = 800  # Match VGGSound training config
  config.dataset_configs.spec_stride = 1

  # These statistics were calculated over the entire unbalanced train set.
  config.dataset_configs.spec_mean = 1.102
  config.dataset_configs.spec_stddev = 2.762

  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.spec_shape = (100, 128)

  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True
  config.dataset_configs.log_test_epochs = 4
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size
  config.dataset_configs.num_test_clips = 4
  config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
  config.multicrop_clips_per_device = 2
  # Leaving this empty means that a full test is done each time.
  # About 4200 / 4 = 1050 steps on a 4-host setting (ie 4x4 TPU)
  # config.steps_per_test = 1000  # Number of test steps taken by each host.

  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = True
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  config.dataset_configs.prefetch_to_device = 2

  # SpecAugment hyperparameters
  config.dataset_configs.spec_augment = True
  config.dataset_configs.spec_augment_params = ml_collections.ConfigDict()
  config.dataset_configs.spec_augment_params.freq_mask_max_bins = 48
  config.dataset_configs.spec_augment_params.freq_mask_count = 1
  config.dataset_configs.spec_augment_params.time_mask_max_frames = 48
  config.dataset_configs.spec_augment_params.time_mask_count = 4
  config.dataset_configs.spec_augment_params.time_warp_max_frames = 1.0
  config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
  config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0

  # Model: MBT-base
  config.model_name = 'mbt_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  # Supports 'rgb' and 'spectrogram'
  config.model.modality_fusion = ('spectrogram', 'rgb')
  config.model.use_bottleneck = True
  config.model.test_with_bottlenecks = True
  config.model.share_encoder = False
  config.model.n_bottlenecks = 4
  
  # Layer at which to fuse. '0' refers to early fusion, if fusion_layer is equal
  # to model.num_layers, then there is no cross-modal attention in the transformer
  # and CLS tokens for each modality are averaged right at the end.
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
  # 3d_conv is only used for RGB inputs.
  config.model.temporal_encoding_config.method = '3d_conv'
  # 32 frames for RGB. Conv filter is 8. So total of 4 frames at input
  config.model.patches.size = [16, 16, 2]
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  config.model.temporal_encoding_config.n_sampled_frames = 4  # Unused here.

  # Inference/analysis specific settings
  config.analysis_mode = True  # Flag to enable activation extraction
  config.save_activations = True
  config.save_attention_weights = True
  config.output_dir = 'analysis_outputs'  # Where to save activations
  
  # Training settings (not used for inference, but required by framework)
  config.trainer_name = 'mbt_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = 0.0
  config.num_training_epochs = 1  # Just 1 pass through data
  config.batch_size = 1  # Process one sample at a time for detailed analysis
  config.rng_seed = 0
  
  # Disable augmentations for inference
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.0  # No mixup during inference
  config.mixmod = False
  config.model.stochastic_droplayer_rate = 0.0  # No stochastic depth during inference

  # Use ImageNet-21k-initialised model from big_vision checkpoint
  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_config = None
 
  # Point to your trained checkpoint for inference
  config.init_from.checkpoint_path = 'mbt_base'  # Path to trained model checkpoint
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.model_config = ml_collections.ConfigDict()
  config.init_from.model_config.model = ml_collections.ConfigDict()
  config.init_from.model_config.model.classifier = 'gap'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'resize_tile'

  # Learning rate (not used for inference, but required)
  steps_per_epoch = AUDIOSET_TEST_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'constant'
  config.lr_configs.base_learning_rate = 0.0  # No learning during inference

  # Logging
  config.write_summary = False  # No tensorboard logging needed
  config.checkpoint = False  # Don't save checkpoints during inference
  config.debug_train = False
  config.debug_eval = False
  return config


