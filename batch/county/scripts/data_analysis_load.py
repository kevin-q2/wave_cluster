import sys
import numpy as np 
import pandas as pd
import networkx as nx
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz

sys.path.append("../wave_cluster")
import wave_cluster as wc


sys.path.append("batch/county/scripts/")
from data_load import load_data


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


county_centers = pd.read_csv("data/us_county_centers.txt")
county_centers_dict = {}

for i in data.columns:
    state_id = int(i[6:8])
    county_id = int(i[8:11])
    
    state_cc = county_centers.loc[county_centers.STATEFP == state_id]
    county_find = state_cc.loc[county_centers.COUNTYFP == county_id]
    try:
        county_loc = (county_find.LATITUDE.values[0], county_find.LONGITUDE.values[0])
        county_centers_dict[i] = county_loc
    except:
        print(i)
        county_centers_dict[i] = (np.inf, np.inf)



def miles_dist(loc1, loc2):
    loc1 = loc1[:loc1.rfind('_')]
    loc2 = loc2[:loc2.rfind('_')]
    x_loc = county_centers_dict[loc1]
    y_loc = county_centers_dict[loc2]
    return wc.haversine(x_loc, y_loc)


try:
    unimodal_cuts = pd.read_csv("batch/county/data/unimodal_cuts.csv", index_col = 0)
    unimodal_cuts = unimodal_cuts.iloc[::2,:]
    unimodal_cuts = unimodal_cuts.loc[:, locations]
    
    #D_uni = load_npz('batch/county/data/northeast_unimodal_pairwise.npz')
    D_uni = load_npz('batch/county/data/east_unimodal_pairwise.npz')
    wpool_uni = wc.wave_pool(data, unimodal_cuts)
    D_uni_miles = wc.pairwise_from_pool(wpool_uni, miles_dist)

except:
    print('Unimodal segmentations not computed yet!')
    pass