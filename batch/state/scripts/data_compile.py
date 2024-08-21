import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz


# Ok this script is a mess but I just use it to read + compile some output results

f1 = pd.read_csv('../data/random_segment_clusters_1.csv', index_col = 0)
f2 = pd.read_csv('../data/random_segment_clusters_2.csv', index_col = 0)
f3 = pd.read_csv('../data/random_segment_clusters_3.csv', index_col = 0)
f4 = pd.read_csv('../data/random_segment_clusters_4.csv', index_col = 0)
f5 = pd.read_csv('../data/random_segment_clusters_5.csv', index_col = 0)
f6 = pd.read_csv('../data/random_segment_clusters_6.csv', index_col = 0)
f7 = pd.read_csv('../data/random_segment_clusters_7.csv', index_col = 0)
f8 = pd.read_csv('../data/random_segment_clusters_8.csv', index_col = 0)
f9 = pd.read_csv('../data/random_segment_clusters_9.csv', index_col = 0)
f10 = pd.read_csv('../data/random_segment_clusters_10.csv', index_col = 0)
F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
F.index = range(len(F))
F.to_csv('../data/random_segment_clusters.csv')



f1 = pd.read_csv('../data/random_segment_random_clusters_1.csv', index_col = 0)
f2 = pd.read_csv('../data/random_segment_random_clusters_2.csv', index_col = 0)
f3 = pd.read_csv('../data/random_segment_random_clusters_3.csv', index_col = 0)
f4 = pd.read_csv('../data/random_segment_random_clusters_4.csv', index_col = 0)
f5 = pd.read_csv('../data/random_segment_random_clusters_5.csv', index_col = 0)
f6 = pd.read_csv('../data/random_segment_random_clusters_6.csv', index_col = 0)
f7 = pd.read_csv('../data/random_segment_random_clusters_7.csv', index_col = 0)
f8 = pd.read_csv('../data/random_segment_random_clusters_8.csv', index_col = 0)
f9 = pd.read_csv('../data/random_segment_random_clusters_9.csv', index_col = 0)
f10 = pd.read_csv('../data/random_segment_random_clusters_10.csv', index_col = 0)
F = pd.concat([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])
F.index = range(len(F))
F.to_csv('../data/random_segment_random_clusters.csv')
