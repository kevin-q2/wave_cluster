import sys
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
import multiprocessing as mp
import scipy
from scipy.sparse import save_npz

sys.path.append("batch/country/scripts/")
from data_load import load_data
sys.path.append("../wave_cluster")
import wave_cluster as wc

cpu_count = 16

# State data
data = load_data()

EU = ['AD', 'AL', 'AM','AT','AZ','BA', 'BE', 'BG','BY', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
     'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 
     'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'GE', 'IS', 'KZ', 'LI', 'MD', 'MC', 'ME',
     'MK', 'NO', 'RU', 'SM', 'RS', 'CH', 'TR', 'UA']

locations = np.sort(EU)
data = data.loc[:, locations]
# These european locations don't have associated containment health data
data = data.drop(['AM','MK', 'ME'], axis = 1)
locations = data.columns


# containment_health
containment_health = pd.read_csv('data/country_containment_health.csv', index_col = 0)
containment_health = containment_health.loc[data.index,:]
data = containment_health


# select segmentations to use
unimodal_cuts = pd.read_csv("batch/country/data/unimodal_cuts.csv", index_col = 0)
unimodal_cuts = unimodal_cuts.iloc[::2,:]

sir_cuts = pd.read_csv('batch/country/data/sir_cuts.csv', index_col = 0)

wavefinder_cuts = pd.read_csv("batch/country/data/wav_cuts.csv", index_col = 0)



D,wpool = wc.compute_pairwise(data, unimodal_cuts.loc[:,locations], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/country/data/unimodal_pairwise_health.npz', D)

D,wpool = wc.compute_pairwise(data, sir_cuts.loc[:,locations], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/country/data/sir_pairwise_health.npz', D)

D,wpool = wc.compute_pairwise(data, wavefinder_cuts.loc[:,locations], wc.align_distance, cpu_count = cpu_count)
save_npz('batch/country/data/wav_pairwise_health.npz', D)

