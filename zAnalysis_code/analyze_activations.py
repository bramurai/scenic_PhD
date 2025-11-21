#!/usr/bin/env python3
"""Analyze extracted activations: PCA, attention flow, etc.

Usage:
  python analyze_activations.py \
    --activation_dir=activation_analysis \
    --output_dir=pca_results
"""

import os
import glob
from absl import app, flags, logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string('activation_dir', 'activation_analysis', 'Directory with extracted activations')
flags.DEFINE_string('output_dir', 'pca_results', 'Output directory for analysis')
flags.DEFINE_integer('n_components', 50, 'Number of PCA components')
flags.DEFINE_bool('run_tsne', False, 'Whether to run t-SNE (slow)')


def load_all_activations(activation_dir: str):
  """Load all activation files."""
  logging.info('Loading activations...')
  
  files = sorted(glob.glob(os.path.join(activation_dir, 'sample_*.npz')))
  logging.info(f'Found {len(files)} activation files')
  
  # Load first file to get layer names
  first_data = np.load(files[0])
  layer_names = [k for k in first_data.keys() if k.startswith('activation_')]
  attention_names = [k for k in first_data.keys() if k.startswith('attention_')]
  
  logging.info(f'Found {len(layer_names)} activation layers')
  logging.info(f'Found {len(attention_names)} attention layers')
  
  # Load all activations
  activations_by_layer = {name: [] for name in layer_names}
  attentions_by_layer = {name: [] for name in attention_names}
  all_logits = []
  
  for file_path in files:
    data = np.load(file_path)
    
    # Load activations
    for name in layer_names:
      activations_by_layer[name].append(data[name])
    
    # Load attention weights
    for name in attention_names:
      attentions_by_layer[name].append(data[name])
    
    # Load logits
    all_logits.append(data['logits'])
  
  # Convert to arrays
  for name in layer_names:
    activations_by_layer[name] = np.array(activations_by_layer[name])
  
  for name in attention_names:
    attentions_by_layer[name] = np.array(attentions_by_layer[name])
  
  all_logits = np.array(all_logits)
  
  return activations_by_layer, attentions_by_layer, all_logits


def run_pca_analysis(activations_by_layer, output_dir, n_components=50):
  """Run PCA on each layer's activations."""
  logging.info(f'\nRunning PCA with {n_components} components...')
  
  pca_results = {}
  
  for layer_name, activations in activations_by_layer.items():
    logging.info(f'  Processing {layer_name}: shape {activations.shape}')
    
    # Flatten spatial dimensions but keep batch and feature dims
    # Shape is typically: (batch, time/space, features)
    original_shape = activations.shape
    
    # Reshape to (batch, -1) for PCA
    batch_size = activations.shape[0]
    flattened = activations.reshape(batch_size, -1)
    
    logging.info(f'    Flattened to: {flattened.shape}')
    
    # Run PCA
    n_comp = min(n_components, flattened.shape[1], flattened.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(flattened)
    
    pca_results[layer_name] = {
        'transformed': transformed,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'original_shape': original_shape,
        'n_components': n_comp
    }
    
    logging.info(f'    Variance explained by first 5 components: '
                f'{pca.explained_variance_ratio_[:5].sum():.3f}')
  
  # Save PCA results
  pca_path = os.path.join(output_dir, 'pca_results.pkl')
  with open(pca_path, 'wb') as f:
    pickle.dump(pca_results, f)
  
  logging.info(f'\nPCA results saved to {pca_path}')
  
  # Plot variance explained
  plot_variance_explained(pca_results, output_dir)
  
  return pca_results


def plot_variance_explained(pca_results, output_dir):
  """Plot cumulative variance explained for each layer."""
  plt.figure(figsize=(15, 10))
  
  for i, (layer_name, result) in enumerate(pca_results.items()):
    plt.subplot(3, 4, i+1)
    plt.plot(result['cumulative_variance'])
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title(layer_name.replace('activation_', '')[:30])
    plt.grid(True)
    
    if i >= 11:  # Limit to 12 subplots
      break
  
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, 'variance_explained.png'), dpi=150)
  logging.info('Saved variance explained plot')


def analyze_attention_flow(attentions_by_layer, output_dir):
  """Analyze attention weight patterns."""
  logging.info('\nAnalyzing attention flow...')
  
  if not attentions_by_layer:
    logging.warning('No attention weights found!')
    return
  
  attention_summary = {}
  
  for layer_name, attention_weights in attentions_by_layer.items():
    logging.info(f'  {layer_name}: shape {attention_weights.shape}')
    
    # Attention weights typically have shape: (batch, num_heads, seq_len, seq_len)
    # Average over batch and heads to get (seq_len, seq_len)
    if len(attention_weights.shape) == 4:
      avg_attention = attention_weights.mean(axis=(0, 1))
    else:
      avg_attention = attention_weights.mean(axis=0)
    
    attention_summary[layer_name] = {
        'average_attention': avg_attention,
        'shape': attention_weights.shape,
        'mean': attention_weights.mean(),
        'std': attention_weights.std()
    }
  
  # Save attention summary
  attention_path = os.path.join(output_dir, 'attention_summary.pkl')
  with open(attention_path, 'wb') as f:
    pickle.dump(attention_summary, f)
  
  logging.info(f'Attention summary saved to {attention_path}')
  
  # Plot attention maps
  plot_attention_maps(attention_summary, output_dir)
  
  return attention_summary


def plot_attention_maps(attention_summary, output_dir):
  """Plot average attention maps."""
  plt.figure(figsize=(15, 10))
  
  for i, (layer_name, summary) in enumerate(attention_summary.items()):
    if i >= 12:  # Limit to 12 plots
      break
    
    plt.subplot(3, 4, i+1)
    plt.imshow(summary['average_attention'], cmap='viridis')
    plt.colorbar()
    plt.title(layer_name.replace('attention_', '')[:30])
  
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, 'attention_maps.png'), dpi=150)
  logging.info('Saved attention maps plot')


def main(argv):
  del argv
  
  logging.info('='*80)
  logging.info('Activation Analysis')
  logging.info('='*80)
  
  # Create output directory
  os.makedirs(FLAGS.output_dir, exist_ok=True)
  
  # Load activations
  activations_by_layer, attentions_by_layer, logits = load_all_activations(FLAGS.activation_dir)
  
  # Run PCA
  pca_results = run_pca_analysis(activations_by_layer, FLAGS.output_dir, FLAGS.n_components)
  
  # Analyze attention
  attention_summary = analyze_attention_flow(attentions_by_layer, FLAGS.output_dir)
  
  # Run t-SNE if requested
  if FLAGS.run_tsne:
    logging.info('\nRunning t-SNE (this may take a while)...')
    # t-SNE on first layer's PCA-reduced features
    first_layer = list(pca_results.keys())[0]
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(pca_results[first_layer]['transformed'][:, :50])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    plt.title('t-SNE of First Layer Activations')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(os.path.join(FLAGS.output_dir, 'tsne.png'), dpi=150)
    logging.info('Saved t-SNE plot')
  
  logging.info('\n' + '='*80)
  logging.info('Analysis Complete!')
  logging.info(f'Results saved to: {FLAGS.output_dir}')
  logging.info('='*80)


if __name__ == '__main__':
  app.run(main)
