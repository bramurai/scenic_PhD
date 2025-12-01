import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform

# Load both RDMs
semantic_data = np.load('Semantic_RDMs/semantic_rdm_all_classes.npz', allow_pickle=True)
semantic_rdm = semantic_data['rdm']
semantic_class_ids = semantic_data['class_ids']

neural_data = np.load('RDM_from_averaged/rdm_encoder_block_L0_audio.npz', allow_pickle=True)
neural_rdm = neural_data['rdm']
neural_class_indices = neural_data['class_indices']  # indices into labels_df

# Map neural indices to mids for alignment
labels_df = pd.read_csv('Video_csvs/audioset_labels.csv')
index_to_mid = dict(zip(labels_df['index'], labels_df['mid']))
neural_mids = [index_to_mid[idx] for idx in neural_class_indices]

# Find common classes
common_mids = np.intersect1d(semantic_class_ids, neural_mids)

# Reindex both RDMs to common classes
semantic_idx = [list(semantic_class_ids).index(mid) for mid in common_mids]
neural_idx = [neural_mids.index(mid) for mid in common_mids]

semantic_rdm_aligned = semantic_rdm[np.ix_(semantic_idx, semantic_idx)]
neural_rdm_aligned = neural_rdm[np.ix_(neural_idx, neural_idx)]

# Compute correlation
rdm_vec_semantic = squareform(semantic_rdm_aligned)
rdm_vec_neural = squareform(neural_rdm_aligned)
corr, pval = spearmanr(rdm_vec_semantic, rdm_vec_neural)
print(f"RDM correlation: r={corr:.4f}, p={pval:.4e}")