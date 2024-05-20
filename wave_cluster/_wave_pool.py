import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
import multiprocessing as mp
from multiprocessing import Pool
from ._tools import overlap_graph



################################################################################################################################
#   Computing pairwise distances between wave segments
#   - Kevin Quinn 11/23
#
#   The Wave Pool Object: 
#   INPUT:
#       data -- (m x n) data matrix where each of the n columns has s segments
#       cuts_table -- (s x n) matrix recording start/end points of each segment for each column of the data
#       distance -- distance function taking inputs (x,y) and computing the distance between x and y using 
#                   some perscribed distance method
#
#   ATTRIBUTES/METHODS:
#       waves -- a dictionary of segments where waves['col_name_i'] gives the ith segment of data[:,col_name]
#       times -- a dictionary of corresponding segment times where times['col_name_i'] = (t1,t2)
#       dist(locations) -- takes a tuple of location names ('col_name_i', 'col_name_j') and computes the 
#                         pairwise distance between locations 
#
#
#   Compute all distances in parallel with compute_pairwise():
#   INPUT:
#       data, cuts_table, distance -- all directly passed into the wave_pool object (same interpretations as above)
#       cpu_count -- number of processors to use in parallel
#
#   OUTPUT:
#       D -- computed pairwise distance matrix (NOTE that I assume symmetric distance function so I record distances
#           only within the upper triangle of the matrix and place these within a sparse csr array)
#       wpool -- the associated wave_pool object created from the data
# 
#
#############################################################################################################################




class wave_pool:
    def __init__(self, data, cuts_table, distance = np.linalg.norm):
        self.data = data
        self.cuts_table = cuts_table
        self.distance = distance
        self.locations = data.columns
        self.waves = {}
        self.times = {}
        self.pool()
        self.key_list = sorted(list(self.waves.keys()))
        self.n = len(self.key_list)
        
    def pool(self):
        for r in range(self.cuts_table.shape[0] - 1):
            for c in range(self.cuts_table.shape[1]):
                if not np.isnan(self.cuts_table.iloc[r,c]) and not np.isnan(self.cuts_table.iloc[r+1,c]):
                    t1 = int(self.cuts_table.iloc[r,c])
                    t2 = int(self.cuts_table.iloc[r+1,c])
                    wv = self.data.iloc[t1:t2,c].to_numpy().flatten()
                    ind = self.locations[c] + '_' + str(r)
                    self.waves[ind] = wv
                    self.times[ind] = (t1,t2)
    
    def dist(self, locations):
        location1 = locations[0]
        location2 = locations[1]
        #times1 = self.times[location1]
        #times2 = self.times[location2]
        #time_intersect = (max(times1[0], times2[0]), min(times1[1], times2[1]))
        #time_union = (min(times1[0], times2[0]), max(times1[1], times2[1]))
        
        x = self.waves[location1]
        y = self.waves[location2]
        #x = self.data.loc[:,location1[:-2]].to_numpy()[time_union[0]:time_union[1]]
        #y = self.data.loc[:,location2[:-2]].to_numpy()[time_union[0]:time_union[1]]
        d = self.distance(x, y)
        return d
    
    
    def save_key_list(self, fname):
        with open(fname, 'w') as f:
            for line in self.key_list:
                f.write(line + "\n")
    
    



def compute_pairwise(data, cuts_table, distance, cpu_count = mp.cpu_count() - 1):
    wpool = wave_pool(data, cuts_table, distance)
    n = len(wpool.key_list)
    idx_entries = []
    d_entries = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            idx_entries.append((i,j))
            d_entries.append((wpool.key_list[i],wpool.key_list[j]))


    with Pool(cpu_count) as p:
        results = p.map(wpool.dist, d_entries)

    row_idx = [i[0] for i in idx_entries]
    col_idx = [i[1] for i in idx_entries]

    # record results in sparse csr matrix
    D = csr_matrix((results, (row_idx, col_idx)), shape = (n,n))
    return D, wpool



def compute_threshold_pairwise(data, cuts_table, distance, percent_overlap, cpu_count = mp.cpu_count() - 1):
    wpool = wave_pool(data, cuts_table, distance)
    edge_list = overlap_graph(percent_overlap, wpool)
    n = len(wpool.key_list)
    idx_entries = []
    d_entries = []
    for e in edge_list:
        i = min(e)
        j = max(e)
        idx_entries.append((i,j))
        d_entries.append((wpool.key_list[i],wpool.key_list[j]))


    with Pool(cpu_count) as p:
        results = p.map(wpool.dist, d_entries)

    row_idx = [i[0] for i in idx_entries]
    col_idx = [i[1] for i in idx_entries]

    # record results in sparse csr matrix
    D = csr_matrix((results, (row_idx, col_idx)), shape = (n,n))
    return D, wpool



def compute_partial_pairwise(pool, idx_entries, cpu_count = mp.cpu_count() - 1):
    n = len(pool.key_list)
    d_entries = []
    for i in range(len(idx_entries)):
        first = idx_entries[i][0]
        second = idx_entries[i][1]
        d_entries.append((pool.key_list[first],pool.key_list[second]))
            
    with Pool(cpu_count) as p:
        results = p.map(pool.dist, d_entries)

    row_idx = [i[0] for i in idx_entries]
    col_idx = [i[1] for i in idx_entries]

    # record results in sparse csr matrix
    D = csr_matrix((results, (row_idx, col_idx)), shape = (n,n))
    return D