
# Minimal TensorFlow-only video augmentation function
# Original Scenic video_ops.py is backed up in video_ops_backup.py

import tensorflow as tf

def basic_video_augment(frames, crop_size=224, is_training=True, zero_centering=True):
    """
    Basic video augmentation using TensorFlow ops.
    Args:
        frames: [num_frames, height, width, channels]
        crop_size: int, output size for crop/resize
        is_training: bool, whether to apply random augmentations
        zero_centering: bool, whether to scale output to [-1, 1]
    Returns:
        Augmented frames tensor
    """
    if is_training:
        frames = tf.image.random_crop(frames, [tf.shape(frames)[0], crop_size, crop_size, 3])
        frames = tf.image.random_flip_left_right(frames)
        frames = tf.image.random_brightness(frames, max_delta=0.2)
        frames = tf.image.random_contrast(frames, lower=0.8, upper=1.2)
    else:
        frames = tf.image.resize_with_crop_or_pad(frames, crop_size, crop_size)
    frames = tf.clip_by_value(frames, 0.0, 1.0)
    if zero_centering:
        frames = frames * 2.0 - 1.0
    return frames
