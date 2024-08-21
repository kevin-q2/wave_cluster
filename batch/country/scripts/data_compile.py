import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz


# Ok this script is a mess but I just use it to read + compile some output results



f1 = pd.read_csv('../data/random_segment_clusters3_1.csv', index_col = 0)
f2 = pd.read_csv('../data/random_segment_clusters3_2.csv', index_col = 0)
f3 = pd.read_csv('../data/random_segment_clusters3_3.csv', index_col = 0)
f4 = pd.read_csv('../data/random_segment_clusters3_4.csv', index_col = 0)
f5 = pd.read_csv('../data/random_segment_clusters3_5.csv', index_col = 0)
f6 = pd.read_csv('../data/random_segment_clusters3_6.csv', index_col = 0)
f7 = pd.read_csv('../data/random_segment_clusters3_7.csv', index_col = 0)
f8 = pd.read_csv('../data/random_segment_clusters3_8.csv', index_col = 0)
f9 = pd.read_csv('../data/random_segment_clusters3_9.csv', index_col = 0)
f10 = pd.read_csv('../data/random_segment_clusters3_10.csv', index_col = 0)
F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
F.index = range(len(F))
F.to_csv('../data/random_segment_clusters3.csv')





f1 = pd.read_csv('../data/random_segment_random_clusters3_1.csv', index_col = 0)
f2 = pd.read_csv('../data/random_segment_random_clusters3_2.csv', index_col = 0)
f3 = pd.read_csv('../data/random_segment_random_clusters3_3.csv', index_col = 0)
f4 = pd.read_csv('../data/random_segment_random_clusters3_4.csv', index_col = 0)
f5 = pd.read_csv('../data/random_segment_random_clusters3_5.csv', index_col = 0)
f6 = pd.read_csv('../data/random_segment_random_clusters3_6.csv', index_col = 0)
f7 = pd.read_csv('../data/random_segment_random_clusters3_7.csv', index_col = 0)
f8 = pd.read_csv('../data/random_segment_random_clusters3_8.csv', index_col = 0)
f9 = pd.read_csv('../data/random_segment_random_clusters3_9.csv', index_col = 0)
f10 = pd.read_csv('../data/random_segment_random_clusters3_10.csv', index_col = 0)
F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
F.index = range(len(F))
F.to_csv('../data/random_segment_random_clusters3.csv')

'''
f1 = pd.read_csv('../data/county_unimodal_cuts1.csv', index_col = 0)
f2 = pd.read_csv('../data/county_unimodal_cuts2.csv', index_col = 0)
f3 = pd.read_csv('../data/county_unimodal_cuts3.csv', index_col = 0)
f4 = pd.read_csv('../data/county_unimodal_cuts4.csv', index_col = 0)
f5 = pd.read_csv('../data/county_unimodal_cuts5.csv', index_col = 0)
f6 = pd.read_csv('../data/county_unimodal_cuts6.csv', index_col = 0)
f7 = pd.read_csv('../data/county_unimodal_cuts7.csv', index_col = 0)
f8 = pd.read_csv('../data/county_unimodal_cuts8.csv', index_col = 0)
f9 = pd.read_csv('../data/county_unimodal_cuts9.csv', index_col = 0)
f10 = pd.read_csv('../data/county_unimodal_cuts10.csv', index_col = 0)
f11 = pd.read_csv('../data/county_unimodal_cuts11.csv', index_col = 0)
f12 = pd.read_csv('../data/county_unimodal_cuts12.csv', index_col = 0)
f13 = pd.read_csv('../data/county_unimodal_cuts13.csv', index_col = 0)
f14 = pd.read_csv('../data/county_unimodal_cuts14.csv', index_col = 0)
f15 = pd.read_csv('../data/county_unimodal_cuts15.csv', index_col = 0)
f16 = pd.read_csv('../data/county_unimodal_cuts16.csv', index_col = 0)
F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16], axis = 1)
F.index = range(len(F))
F.to_csv('../data/county_unimodal_cuts.csv')
'''

'''
D1 = load_npz('../data/country/unimodal_pairwise_1.npz')
D2 = load_npz('../data/country/unimodal_pairwise_2.npz')
D3 = load_npz('../data/country/unimodal_pairwise_3.npz')
D4 = load_npz('../data/country/unimodal_pairwise_4.npz')
D5 = load_npz('../data/country/unimodal_pairwise_5.npz')
D = D1 + D2 + D3 + D4 + D5
save_npz('../data/country/unimodal_pairwise.npz', D)
'''


'''
sir_waves1 = pd.read_csv("batch/data/state_sir_waves1.csv", index_col = 0)
sir_cuts1 = pd.read_csv("batch/data/state_sir_cuts1.csv", index_col = 0)

sir_waves2 = pd.read_csv("batch/data/state_sir_waves2.csv", index_col = 0)
sir_cuts2 = pd.read_csv("batch/data/state_sir_cuts2.csv", index_col = 0)

sir_waves3 = pd.read_csv("batch/data/state_sir_waves3.csv", index_col = 0)
sir_cuts3 = pd.read_csv("batch/data/state_sir_cuts3.csv", index_col = 0)

sir_waves4 = pd.read_csv("batch/data/state_sir_waves4.csv", index_col = 0)
sir_cuts4 = pd.read_csv("batch/data/state_sir_cuts4.csv", index_col = 0)

sir_cuts = sir_cuts1.join([sir_cuts2, sir_cuts3, sir_cuts4])
sir_waves = sir_waves1.join([sir_waves2, sir_waves3, sir_waves4])

sir_waves.to_csv('batch/data/state/state_sir_waves.csv')
sir_cuts.to_csv('batch/data/state/state_sir_cuts.csv')
'''