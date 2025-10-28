# Copyright 2025 The Scenic Authors - GPU Adapted Version
#
# This is a GPU-adapted version of trainer.py
# Key changes:
# - Support for both single GPU (jit) and multi-GPU (pmap)
# - Automatic device detection
# - Conditional replication based on device count

"""Training Script for MBT - GPU Compatible Version."""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.mbt import train_utils as mbt_train_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils

from scenic.train_lib import lr_schedules 
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def get_device_count():
  """Get the number of available GPUs or fallback to CPU."""
  try:
    devices = jax.devices('gpu')
    logging.info(f'Found {len(devices)} GPU(s): {devices}')
    return len(devices)
  except RuntimeError:
    logging.warning('No GPUs found, falling back to CPU')
    return 1


def is_multi_device():
  """Check if we're running on multiple devices."""
  return get_device_count() > 1


# Import the mixup function from original trainer
def mixup_modalities(batch: Dict['str', Any],
                     alpha: float = 1.0,
                     batch_first: bool = True,
                     mixmod: bool = False,
                     rng: Optional[Any] = None) -> Dict['str', jnp.ndarray]:
  """Mixes multimodal inputs and labels within a single batch.
  
  NOTE: This function is unchanged from the original trainer.py
  """
  inputs, labels = batch['inputs'], batch['label']
  batch['label'] = {}
  num_modalities = len(inputs)

  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')

  batch_size = labels.shape[0]

  # Setup the the numpy backend and prepare mixup weights.
  if rng is None:
    np_backend = np  # Ordinary numpy
    if mixmod:
      weights = list(np_backend.random.beta(alpha, alpha, size=num_modalities))
    else:
      weights = [np_backend.random.beta(alpha, alpha)] * num_modalities
  else:
    np_backend = jnp  # JAX numpy
    if mixmod:
      weights = list(jax.random.beta(rng, alpha, alpha, shape=[num_modalities]))
    else:
      weights = [jax.random.beta(rng, alpha, alpha)] * num_modalities
  for i in range(num_modalities):
    weights[i] *= np_backend.ones((batch_size, 1))

  # Mixup inputs.
  for modality, values in inputs.items():
    weight = weights[len(batch['label'])]
    # Mixup labels.
    batch['label'][modality] = weight * labels + (1.0 - weight) * labels[::-1]
    weight_shape = np.ones((values.ndim))
    if batch_first:
      weight_shape[0] = batch_size
    else:
      weight_shape[-1] = batch_size
    weight = np_backend.reshape(weight,
                                weight_shape.astype(np_backend.int32))
    reverse = []
    for i in range(values.ndim):
      if (i == 0 and batch_first) or (i == values.ndim - 1 and not batch_first):
        reverse.append(slice(-1, None, -1))
      else:
        reverse.append(slice(values.shape[i]))
    batch['inputs'][modality] = (weight * values +
                                 (1.0 - weight) * values[tuple(reverse)])
  if num_modalities == 1 or not mixmod:
    batch['label']['all'] = weights[0] * labels + (1.0 -
                                                   weights[0]) * labels[::-1]

  return batch


def train_step(
  train_state: train_utils.TrainState,
  batch: Batch,
  *,
  flax_model: nn.Module,
  learning_rate_fn: Callable[[int], float],
  loss_fn: LossFn,
  metrics_fn: MetricFn,
  config: ml_collections.ConfigDict,
  optimizer: Any,
  debug: Optional[bool] = False,
  use_pmap: bool = True  # NEW: control whether to use pmap-specific code
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], float]:
  """Runs a single step of training.
  
  GPU adaptation: Added use_pmap parameter to control cross-device ops.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Mixup.
  if config.get('mixup') and config.mixup.alpha:
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_modality = config.get('mixmod_mixup', False)
    batch = mixup_modalities(
        batch,
        config.mixup.alpha,
        batch_first=True,
        mixmod=mixup_modality,
        rng=mixup_rng)
  else:
    labels = batch['label']
    batch['label'] = {}
    for modality in batch['inputs']:
      batch['label'][modality] = labels
    batch['label']['all'] = labels

  # Bind the rng to the host/device we are on for dropout.
  # GPU adaptation: This works for both single and multi-GPU
  if use_pmap:
    dropout_rng = train_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to='device')
  else:
    dropout_rng = rng

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.params)

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  del train_cost
  
  # GPU adaptation: Only use pmean when running on multiple devices
  if use_pmap:
    grad = jax.lax.pmean(grad, axis_name='batch')
  
  updates, new_opt_state = optimizer.update(grad, train_state.opt_state, train_state.params)
  new_params = optimizers.optax.apply_updates(train_state.params, updates)

  # Explicit weight decay, if necessary.
  if config.get('explicit_weight_decay', None) is not None:
    new_params = optimizers.tree_map_with_names(
        functools.partial(
            optimizers.decay_weight_fn,
            lr=lr,
            decay=config.explicit_weight_decay),
        new_params,
        match_name_fn=lambda name: 'kernel' in name)

  if isinstance(logits, dict):
    modality = list(logits.keys())[0]
    batch['label'] = batch['label'][modality]
    metrics = metrics_fn(logits[modality], batch)
  else:
    metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(
      global_step=step + 1,
      params=new_params,
      opt_state=new_opt_state,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr


# Helper function to conditionally replicate/unreplicate
def maybe_replicate(obj, use_pmap=True):
  """Replicate object if using pmap, otherwise return as-is."""
  if use_pmap:
    return jax_utils.replicate(obj)
  return obj


def maybe_unreplicate(obj, use_pmap=True):
  """Unreplicate object if using pmap, otherwise return as-is."""
  if use_pmap:
    return jax_utils.unreplicate(obj)
  return obj


def maybe_unreplicate_and_get(obj, use_pmap=True):
  """Unreplicate and get if using pmap, otherwise just get."""
  if use_pmap:
    return train_utils.unreplicate_and_get(obj)
  return jax.tree_util.tree_map(lambda x: x.item() if hasattr(x, 'item') else x, obj)


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    return_logits_and_labels: bool = False,
    debug: Optional[bool] = False,
    use_pmap: bool = True  # NEW: control whether to use pmap-specific code
) -> Union[Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray], Dict[
    str, Tuple[float, int]]]:
  """Runs a single step of evaluation.
  
  GPU adaptation: Added use_pmap parameter to control cross-device ops.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer.
    batch: A single batch of data.
    flax_model: A Flax model.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    return_logits_and_labels: Whether to return logits and labels or not.
    debug: Whether the debug mode is enabled during evaluation.
    use_pmap: Whether using pmap (multi-device) or jit (single-device).

  Returns:
    Calculated metrics [and optionally logits].
  """
  variables = {
    'params': train_state.params,
    **train_state.model_state
  }
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      train=False, mutable=False, debug=debug)

  metrics = metrics_fn(logits, batch)
  
  if return_logits_and_labels:
    if use_pmap:
      logits = jax.lax.all_gather(logits, 'batch')
      labels = jax.lax.all_gather(batch['label'], 'batch')
    else:
      # For single device, no gathering needed
      labels = batch['label']
    return metrics, logits, labels
  return metrics


def test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    n_clips: int = 2,
    return_logits_and_labels: bool = False,
    softmax_logits: bool = False,
    debug: bool = False,
    use_pmap: bool = True  # NEW: control whether to use pmap-specific code
) -> Union[
    Dict[str, Tuple[float, int]],
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray],
]:
  """Runs a single step of testing with multi-crop evaluation.
  
  GPU adaptation: Added use_pmap parameter to control cross-device ops.

  For multi-crop testing, we assume that num_crops consecutive entries in the
  batch are from the same example. And we average the logits over these examples.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer, and other metadata.
    batch: Dictionary with keys 'inputs', 'labels', 'batch_mask'.
    flax_model: A Flax model.
    metrics_fn: Metrics function for the model.
    n_clips: The number of clips to process at a time by each device.
    return_logits_and_labels: Whether return logits of the model or not.
    softmax_logits: Whether to softmax-normalise the logits before averaging.
    debug: Whether the debug mode is enabled during evaluation.
    use_pmap: Whether using pmap (multi-device) or jit (single-device).

  Returns:
    Calculated metrics [and optionally averaged logits].
  """
  all_logits = jnp.zeros(batch['label'].shape[1])
  assert len(batch['batch_mask'].shape) == 1, (
      'Spatial padding is not supported in multi-crop evaluation.')

  variables = {
    'params': train_state.params,
    **train_state.model_state
  }
  for modality in batch['inputs']:
    num_crops = batch['inputs'][modality].shape[0]
  for idx in range(0, num_crops, n_clips):
    current_input = {}
    for modality in batch['inputs']:
      current_input[modality] = batch['inputs'][modality][idx:idx + n_clips]
    logits = flax_model.apply(
        variables, current_input, train=False, mutable=False, debug=debug)

    if softmax_logits:
      logits = nn.softmax(logits, axis=-1)
    logits = jnp.sum(logits, axis=0)
    all_logits = all_logits + logits

  all_logits = all_logits / num_crops
  all_logits = jnp.expand_dims(all_logits, axis=0)
  batch['label'] = jnp.expand_dims(batch['label'][0], axis=0)
  batch['batch_mask'] = jnp.expand_dims(batch['batch_mask'][0], axis=0)
  metrics = metrics_fn(all_logits, batch)
  
  if return_logits_and_labels:
    if use_pmap:
      all_logits = jax.lax.all_gather(all_logits, 'batch')
      labels = jax.lax.all_gather(batch['label'], 'batch')
    else:
      labels = batch['label']
    return metrics, all_logits, labels
  return metrics


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop - GPU compatible version.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  GPU adaptations:
  - Automatically detects available GPUs
  - Uses jit for single GPU, pmap for multi-GPU
  - Conditional replication based on device count

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics.
  """
  lead_host = jax.process_index() == 0
  
  # GPU adaptation: Detect device configuration
  num_devices = get_device_count()
  use_pmap = is_multi_device()
  logging.info(f'GPU setup: {num_devices} device(s), using {"pmap" if use_pmap else "jit"}')
  
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  is_multilabel_model = (config.model_name == 'mbt_multilabel_classification')

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  input_shapes = dataset.meta_data['input_shape']
  input_dtype = dataset.meta_data.get('input_dtype', jnp.float32)
  if isinstance(input_shapes, dict):
    input_spec = {
        modality: (input_shapes[modality], input_dtype)
        for modality in input_shapes
    }
  else:
    input_spec = [(input_shapes, input_dtype)]
  (params, model_state, num_trainable_params,
   gflops) = mbt_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  # Create optimizer using Optax (train_lib).
  optimizer = optimizers.get_optimizer(config, learning_rate_fn, params)
  opt_state = optimizer.init(params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
    global_step=0,
    params=params,
    opt_state=opt_state,
    model_state=model_state,
    rng=train_rng,
    accum_train_time=0)
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
    if checkpoint_format == 'scenic':
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
    elif checkpoint_format == 'big_vision':
      restored_train_state = pretrain_utils.convert_big_vision_to_scenic_checkpoint(
          init_checkpoint_path, train_state)
      restored_model_cfg = copy.deepcopy(config)
      restored_model_cfg.model.classifier = config.init_from.get(
          'classifier_type', 'token')

    train_state = model.init_from_train_state(
        train_state, restored_train_state, restored_model_cfg,
        restore_output_proj=config.init_from.get('restore_output_proj', False))
    del restored_train_state
  elif start_step == 0:
    logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')

  # GPU adaptation: Conditional replication
  train_state = maybe_replicate(train_state, use_pmap)
  del params  # Do not keep a copy of the initial params.

  # Compile training and evaluation steps
  if use_pmap:
    # Multi-GPU: use pmap
    train_step_compiled = jax.pmap(
      functools.partial(
        train_step,
        flax_model=model.flax_model,
        learning_rate_fn=learning_rate_fn,
        loss_fn=model.loss_function,
        metrics_fn=model.get_metrics_fn('train'),
        config=config,
        optimizer=optimizer,
        debug=config.debug_train,
        use_pmap=True),
      axis_name='batch',
      donate_argnums=(0, 1),
    )
    eval_step_compiled = jax.pmap(
        functools.partial(
            eval_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('validation'),
            return_logits_and_labels=is_multilabel_model,
            debug=config.debug_eval,
            use_pmap=True),
        axis_name='batch',
        donate_argnums=(1,),
    )
  else:
    # Single GPU: use jit
    train_step_compiled = jax.jit(
      functools.partial(
        train_step,
        flax_model=model.flax_model,
        learning_rate_fn=learning_rate_fn,
        loss_fn=model.loss_function,
        metrics_fn=model.get_metrics_fn('train'),
        config=config,
        optimizer=optimizer,
        debug=config.debug_train,
        use_pmap=False),
    )
    eval_step_compiled = jax.jit(
        functools.partial(
            eval_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('validation'),
            return_logits_and_labels=is_multilabel_model,
            debug=config.debug_eval,
            use_pmap=False),
    )
    
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  log_test_steps = 0
  
  if config.dataset_configs.get('do_multicrop_test'):
    log_test_steps = int(steps_per_epoch *
                         config.dataset_configs.log_test_epochs)

    if use_pmap:
      test_step_compiled = jax.pmap(
          functools.partial(
              test_step,
              flax_model=model.flax_model,
              metrics_fn=model.get_metrics_fn('test'),
              n_clips=config.get('multicrop_clips_per_device', 2),
              return_logits_and_labels=is_multilabel_model,
              debug=config.debug_eval,
              use_pmap=True),
          axis_name='batch',
          donate_argnums=(1,),
      )
    else:
      test_step_compiled = jax.jit(
          functools.partial(
              test_step,
              flax_model=model.flax_model,
              metrics_fn=model.get_metrics_fn('test'),
              n_clips=config.get('multicrop_clips_per_device', 2),
              return_logits_and_labels=is_multilabel_model,
              debug=config.debug_eval,
              use_pmap=False),
      )

    if use_pmap:
      device_count = jax.local_device_count()
    else:
      device_count = 1
      
    assert config.dataset_configs.test_batch_size == device_count, (
        f'The per-host batch size must be equal to the number of local devices. '
        f'Got {config.dataset_configs.test_batch_size} vs {device_count}')

    total_test_steps = int(
        np.ceil(dataset.meta_data['num_test_examples'] /
                (config.get('dataset_configs.test_batch_size') *
                 config.get('dataset_configs.num_test_clips') *
                 jax.process_count())))
    steps_per_test = config.get('steps_per_test') or total_test_steps

  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  # GPU adaptation: Conditional unreplication for chrono
  accum_time = maybe_unreplicate(train_state.accum_train_time, use_pmap)
  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(accum_time))

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, lr = train_step_compiled(train_state, train_batch)
      train_metrics.append(t_metrics)
      extra_training_logs.append({'learning_rate': lr})

    for h in hooks:
      h(step)

    chrono.pause()  # Below are once-in-a-while ops -> pause.
    
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer=writer)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              functools.partial(maybe_unreplicate_and_get, use_pmap=use_pmap),
              train_metrics),
          extra_training_logs=jax.tree_util.tree_map(
              functools.partial(maybe_unreplicate_and_get, use_pmap=use_pmap),
              extra_training_logs),
          writer=writer,
          key_separator='/')
      train_metrics, extra_training_logs = [], []

    ################### EVALUATION ################################
    if (step % log_eval_steps == 1) or (step == total_steps):
      with report_progress.timed('eval'):
        eval_metrics = []
        additional_summary = None
        if is_multilabel_model:
          eval_logits = []
          eval_labels = []
          n_classes = dataset.meta_data['num_classes']
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics = eval_step_compiled(train_state, eval_batch)
          if is_multilabel_model:
            e_metrics, logits_batch, labels_batch = e_metrics
            eval_logits.append(vivit_train_utils.to_cpu(logits_batch))
            eval_labels.append(vivit_train_utils.to_cpu(labels_batch))
          # Fetch e_metrics to host and store.
          eval_metrics.append(
              maybe_unreplicate_and_get(e_metrics, use_pmap))
        if is_multilabel_model:
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(eval_logits, axis=0),
              np.concatenate(eval_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        # Log eval summary.
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=additional_summary,
            writer=writer,
            key_separator='/')
        writer.flush()
        del eval_metrics
        
    ##################### CHECKPOINTING ###########################
    if ((step % checkpoint_steps == 0 and step > 0) or (step == total_steps) or
        (step % log_eval_steps == 1)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    ############# MULTICROP TESTING ############################
    if (config.dataset_configs.get('do_multicrop_test') and
        ((step % log_test_steps == 1 and step > 1) or step == total_steps)):
      with report_progress.timed('test'):
        test_metrics = []
        additional_summary = None
        if is_multilabel_model:
          test_logits = []
          test_labels = []
          n_classes = dataset.meta_data['num_classes']
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)

        # At the end of training, evaluate on the whole test set.
        if step == total_steps:
          steps_per_test = total_test_steps

        logging.info('Starting multicrop test')
        for _ in range(steps_per_test):
          test_batch = next(dataset.test_iter)
          t_metrics = test_step_compiled(train_state, test_batch)
          if is_multilabel_model:
            t_metrics, logits_batch, labels_batch = t_metrics
            test_logits.append(vivit_train_utils.to_cpu(logits_batch))
            test_labels.append(vivit_train_utils.to_cpu(labels_batch))
          # Fetch t_metrics to host and store.
          test_metrics.append(
              maybe_unreplicate_and_get(t_metrics, use_pmap))
        if is_multilabel_model:
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(test_logits, axis=0),
              np.concatenate(test_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        # Log eval summary.
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=test_metrics,
            writer=writer,
            extra_eval_summary=additional_summary,
            prefix='test',
            key_separator='/')
        logging.info('Completed multicrop test')
        writer.flush()
        del test_metrics

    chrono.resume()  # un-pause now
    
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary



