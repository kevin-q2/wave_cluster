import sys
import numpy as np 
import pandas as pd
import networkx as nx
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

sys.path.append("batch/state/scripts/")
from data_load import load_data

sys.path.append("../wave_cluster")
import wave_cluster as wc


# This is a script meant to be an automatic way of loading and pre-processing some of the auxiliary data that 
# we use for analysis

# Load the dataset
data = load_data()
locations = data.columns

# geographic information
us_adj = nx.read_edgelist('data/us_adjacency.txt')
us_edges = list(us_adj.edges)
us_edges = [('US_' + i[0], 'US_' + i[1], 1) for i in us_edges]
state_centers = pd.read_csv("data/state_centers.csv", index_col = 0)
state_centers_dict = {i: np.array(state_centers.loc[state_centers.STNAME == i].iloc[:,1:]).flatten() for i in state_centers.STNAME.values}

# density for each location 
density = pd.read_csv('data/us_state_density.csv', index_col = 0)

# containment_health
containment_health = pd.read_csv('data/state_containment_health.csv', index_col = 0)
containment_health = containment_health.loc[data.index,:]
containment_health = containment_health.loc[:, data.columns]


def miles_dist(loc1, loc2):
    loc1 = loc1[:loc1.rfind('_')]
    loc2 = loc2[:loc2.rfind('_')]
    x_loc = state_centers_dict[loc1]
    y_loc = state_centers_dict[loc2]
    return wc.haversine(x_loc, y_loc)

try:
    unimodal_cuts = pd.read_csv("batch/state/data/unimodal_cuts.csv", index_col = 0)
    unimodal_cuts = unimodal_cuts.iloc[::2,:]
    D_uni = load_npz('batch/state/data/unimodal_pairwise.npz')
    wpool_uni = wc.wave_pool(data, unimodal_cuts[data.columns])
    
    D_uni_miles = wc.pairwise_from_pool(wpool_uni, miles_dist)
    D_uni_health = load_npz('batch/state/data/unimodal_pairwise_health.npz')

except:
    print('Unimodal segmentations not computed yet!')
    pass


try:
    sir_cuts = pd.read_csv('batch/state/data/sir_cuts.csv', index_col = 0)
    D_sir = load_npz('batch/state/data/sir_pairwise.npz')
    wpool_sir = wc.wave_pool(data, sir_cuts[data.columns])
    
    D_sir_miles = wc.pairwise_from_pool(wpool_sir, miles_dist)
    D_sir_health = load_npz('batch/state/data/sir_pairwise_health.npz')

except:
    print('SIR segmentations not computed yet!')
    pass


try:
    wavefinder_cuts = pd.read_csv("batch/state/data/wav_cuts.csv", index_col = 0)
    D_wav = load_npz('batch/state/data/wav_pairwise.npz')
    wpool_wav = wc.wave_pool(data, wavefinder_cuts[data.columns])
    
    D_wav_miles = wc.pairwise_from_pool(wpool_wav, miles_dist)
    D_wav_health = load_npz('batch/state/data/wav_pairwise_health.npz')

except:
    print('WAV segmentations not computed yet!')
    pass