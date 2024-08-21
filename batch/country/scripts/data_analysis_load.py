import sys
import numpy as np 
import pandas as pd
import networkx as nx
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

sys.path.append("batch/country/scripts/")
from data_load import load_data

sys.path.append("wave_cluster")
import wave_cluster as wc

data = load_data()

# chosen subset of european countries
EU = ['AD', 'AL', 'AM','AT','AZ','BA', 'BE', 'BG','BY', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
     'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 
     'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'GE', 'IS', 'KZ', 'LI', 'MD', 'MC', 'ME',
     'MK', 'NO', 'RU', 'SM', 'RS', 'CH', 'TR', 'UA']

locations = np.sort(EU)
data = data.loc[:, locations]
# These european locations don't have associated containment health data
data = data.drop(['AM','MK', 'ME'], axis = 1)
locations = data.columns

## NEED TO INCLUDE GEOGRaphic info and containment health!!
geo = pd.read_csv("data/geography.csv", index_col = 0)
country_centers_dict = {i: np.array(geo.loc[i, ['latitude', 'longitude']]).flatten() for i in data.columns}
# Some locations were un-reported so I use coordinates reported by wikidata
#country_centers_dict['AM'] = np.array([40.383333, 44.95])
#country_centers_dict['MK'] = np.array([41.65, 21.716667])
country_centers_dict['GE'] = np.array([42, 44])

# containment_health
containment_health = pd.read_csv('data/country_containment_health.csv', index_col = 0)
containment_health = containment_health.loc[data.index,:]

def miles_dist(loc1, loc2):
    loc1 = loc1[:loc1.rfind('_')]
    loc2 = loc2[:loc2.rfind('_')]
    x_loc = country_centers_dict[loc1]
    y_loc = country_centers_dict[loc2]
    return wc.haversine(x_loc, y_loc)


try:
    unimodal_cuts = pd.read_csv("batch/country/data/unimodal_cuts.csv", index_col = 0)
    #unimodal_cuts = unimodal_cuts.iloc[::2,:]
    unimodal_cuts = unimodal_cuts.loc[:, locations]
    D_uni = load_npz('batch/country/data/unimodal_pairwise.npz')
    wpool_uni = wc.wave_pool(data, unimodal_cuts)

    D_uni_miles = wc.pairwise_from_pool(wpool_uni, miles_dist)
    D_uni_health = load_npz('batch/country/data/unimodal_pairwise_health.npz')

except:
    print('Unimodal segmentations not computed yet!')
    pass


try:
    sir_cuts = pd.read_csv('batch/country/data/sir_cuts.csv', index_col = 0)
    D_sir = load_npz('batch/country/data/sir_pairwise.npz')
    wpool_sir = wc.wave_pool(data, sir_cuts)
    
    D_sir_miles = wc.pairwise_from_pool(wpool_sir, miles_dist)
    D_sir_health = load_npz('batch/country/data/sir_pairwise_health.npz')

except:
    print('SIR segmentations not computed yet!')
    pass


try:
    wavefinder_cuts = pd.read_csv("batch/country/data/wav_cuts.csv", index_col = 0)
    wavefinder_cuts = wavefinder_cuts.loc[:, locations]
    D_wav = load_npz('batch/country/data/wav_pairwise.npz')
    wpool_wav = wc.wave_pool(data, wavefinder_cuts[data.columns])
    
    D_wav_miles = wc.pairwise_from_pool(wpool_wav, miles_dist)
    D_wav_health = load_npz('batch/country/data/wav_pairwise_health.npz')

except:
    print('WAV segmentations not computed yet!')
    pass