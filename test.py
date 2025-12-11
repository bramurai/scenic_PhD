"""
path = /project/3026018.01/Models/MBT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from absl import app, flags, logging
import pickle   

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint.pkl with metadata')

def load_checkpoint_metadata(checkpoint_path):
  """Load metadata from checkpoint without loading large data."""
  with open(checkpoint_path, 'rb') as f:
    data = pickle.load(f)
  
  return {
      'counts': data['counts'],
      'num_classes': data['num_classes'],
      'activation_names': data.get('activation_names', []),
      'processed_count': data.get('processed_count', 0)
  }
def main(argv):
    del argv  # Unused

    logging.info(f'Loading checkpoint metadata from {FLAGS.checkpoint_path}')
    metadata = load_checkpoint_metadata(FLAGS.checkpoint_path)

    counts = metadata['counts']
    num_classes = metadata['num_classes']
    activation_names = metadata['activation_names']
    processed_count = metadata['processed_count']

    logging.info(f'Number of classes: {num_classes}')
    logging.info(f'Activation names: {activation_names}')
    logging.info(f'Total processed samples: {processed_count}')

 

if __name__ == '__main__':
    app.run(main)