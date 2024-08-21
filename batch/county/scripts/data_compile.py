import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz


# Ok this script is a mess but I just use it to read + compile some output results



f1 = pd.read_csv('../data/northeast_random_segment_1.csv', index_col = 0)
f2 = pd.read_csv('../data/northeast_random_segment_2.csv', index_col = 0)
f3 = pd.read_csv('../data/northeast_random_segment_3.csv', index_col = 0)
f4 = pd.read_csv('../data/northeast_random_segment_4.csv', index_col = 0)
f5 = pd.read_csv('../data/northeast_random_segment_5.csv', index_col = 0)
f6 = pd.read_csv('../data/northeast_random_segment_6.csv', index_col = 0)
f7 = pd.read_csv('../data/northeast_random_segment_7.csv', index_col = 0)
f8 = pd.read_csv('../data/northeast_random_segment_8.csv', index_col = 0)
f9 = pd.read_csv('../data/northeast_random_segment_9.csv', index_col = 0)
f10 = pd.read_csv('../data/northeast_random_segment_10.csv', index_col = 0)

F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
#F = pd.concat([f2,f4,f5,f7,f8,f9])
F.index = range(len(F))
F.to_csv('../data/northeast_random_segment.csv')



f1 = pd.read_csv('../data/northeast_random_segment_random_clusters_1.csv', index_col = 0)
f2 = pd.read_csv('../data/northeast_random_segment_random_clusters_2.csv', index_col = 0)
f3 = pd.read_csv('../data/northeast_random_segment_random_clusters_3.csv', index_col = 0)
f4 = pd.read_csv('../data/northeast_random_segment_random_clusters_4.csv', index_col = 0)
f5 = pd.read_csv('../data/northeast_random_segment_random_clusters_5.csv', index_col = 0)
f6 = pd.read_csv('../data/northeast_random_segment_random_clusters_6.csv', index_col = 0)
f7 = pd.read_csv('../data/northeast_random_segment_random_clusters_7.csv', index_col = 0)
f8 = pd.read_csv('../data/northeast_random_segment_random_clusters_8.csv', index_col = 0)
f9 = pd.read_csv('../data/northeast_random_segment_random_clusters_9.csv', index_col = 0)
f10 = pd.read_csv('../data/northeast_random_segment_random_clusters_10.csv', index_col = 0)

F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
#F = pd.concat([f1,f3,f4,f5,f7])
F.index = range(len(F))
F.to_csv('../data/northeast_random_segment_random_cluster.csv')

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
D1 = load_npz('batch/county/data/east_unimodal_pairwise_1.npz')
D2 = load_npz('batch/county/data/east_unimodal_pairwise_2.npz')
D3 = load_npz('batch/county/data/east_unimodal_pairwise_3.npz')
D4 = load_npz('batch/county/data/east_unimodal_pairwise_4.npz')
D5 = load_npz('batch/county/data/east_unimodal_pairwise_5.npz')
D6 = load_npz('batch/county/data/east_unimodal_pairwise_6.npz')
D7 = load_npz('batch/county/data/east_unimodal_pairwise_7.npz')
D8 = load_npz('batch/county/data/east_unimodal_pairwise_8.npz')
D9 = load_npz('batch/county/data/east_unimodal_pairwise_9.npz')
D10 = load_npz('batch/county/data/east_unimodal_pairwise_10.npz')
D11 = load_npz('batch/county/data/east_unimodal_pairwise_11.npz')
D12 = load_npz('batch/county/data/east_unimodal_pairwise_12.npz')
D13 = load_npz('batch/county/data/east_unimodal_pairwise_13.npz')
D14 = load_npz('batch/county/data/east_unimodal_pairwise_14.npz')
D15 = load_npz('batch/county/data/east_unimodal_pairwise_15.npz')
D16 = load_npz('batch/county/data/east_unimodal_pairwise_16.npz')
D17 = load_npz('batch/county/data/east_unimodal_pairwise_17.npz')
D18 = load_npz('batch/county/data/east_unimodal_pairwise_18.npz')
D19 = load_npz('batch/county/data/east_unimodal_pairwise_19.npz')
D20 = load_npz('batch/county/data/east_unimodal_pairwise_20.npz')
D = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 + D11 + D12 + D13 + D14 + D15 + D16 + D17 + D18 + D19 + D20
save_npz('batch/county/data/east_unimodal_pairwise.npz', D)
'''