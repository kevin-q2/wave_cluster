import sys
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp
import scipy
from scipy.sparse import save_npz

sys.path.append("batch/state/scripts/")
from data_load import load_data
sys.path.append("wave_cluster")
import wave_cluster as wc

cpu_count = 16

# State data
data = load_data()

# containment_health
containment_health = pd.read_csv('data/state_containment_health.csv', index_col = 0)
containment_health = containment_health.loc[data.index,:]
containment_health = containment_health.loc[:,data.columns]
data = containment_health


# select segmentations to use
unimodal_cuts = pd.read_csv("batch/state/data/unimodal_cuts.csv", index_col = 0)
unimodal_cuts = unimodal_cuts.iloc[::2,:]
sir_cuts = pd.read_csv('batch/state/data/sir_cuts.csv', index_col = 0)
wavefinder_cuts = pd.read_csv("batch/state/data/wav_cuts.csv", index_col = 0)


# Computes pairwise segment distance matrices
D,wpool = wc.compute_pairwise(data, unimodal_cuts.loc[:,data.columns], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/state/data/unimodal_pairwise.npz', D)

D,wpool = wc.compute_pairwise(data, sir_cuts.loc[:,data.columns], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/state/data/sir_pairwise.npz', D)

D,wpool = wc.compute_pairwise(data, wavefinder_cuts.loc[:,data.columns], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/state/data/wav_pairwise.npz', D)

