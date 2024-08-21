import sys
import os
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp
import scipy
from scipy.sparse import save_npz

sys.path.append("batch/county/scripts/")
from data_load import load_data
sys.path.append("wave_cluster")
import wave_cluster as wc



cpu_count = 8

# Load in dataset
data = load_data()

# USE for Northeastern US counties
#states = ['NY', 'NJ', 'PA', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME']


# USE for all Eastern US counties
states = ['NY', 'NJ', 'PA', 'CT', 'RI', 'MA', 'VT', 'NH', 'ME',
         'MD', 'DE', 'WV', 'VA', 'KY', 'TN', 'NC',
         'MI', 'WI', 'IL', 'IN', 'OH',
         'SC', 'GA', 'AL', 'MS', 'FL']

locations = data.columns[data.columns.str.contains('|'.join(states))]
data = data.loc[:, locations]


# select segmentations to use
unimodal_cuts = pd.read_csv("batch/county/data/unimodal_cuts.csv", index_col = 0)
unimodal_cuts = unimodal_cuts.iloc[::2,:]
unimodal_cuts = unimodal_cuts.loc[:,locations]
est_cuts = unimodal_cuts


min_overlap = 0.5
wpool = wc.wave_pool(data, est_cuts, wc.align_distance)
edge_list = wc.overlap_graph(min_overlap, wpool)

n = len(wpool.key_list)        
idx_entries = []
d_entries = []
for e in edge_list:
    i = min(e)
    j = max(e)
    idx_entries.append((i,j))
    d_entries.append((wpool.key_list[i],wpool.key_list[j]))
        
        
job_id = int(os.getenv('SGE_TASK_ID'))
max_batches = 20
percent_per_batch = 1/max_batches
per_batch = int(percent_per_batch*len(idx_entries))

if job_id != max_batches:
    batch_idx_entries = idx_entries[per_batch*(job_id - 1):per_batch*(job_id)]
else:
    batch_idx_entries = idx_entries[per_batch*(job_id - 1):]

D = wc.compute_partial_pairwise(wpool, batch_idx_entries, cpu_count = cpu_count)
save_npz('batch/county/data/east_unimodal_pairwise_' + str(job_id) + '.npz', D)

