import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from ._dynamic_time_warp import dtw
from itertools import combinations
from scipy.sparse import csr_matrix


#######################################################################################
# The following functions are an assorted set of important tools that I use in 
# other parts of the package, as well as in my experiments. They 
# are all described individually below. 
#   -Kevin Quinn 1/24
#
########################################################################################



# Take a sliding windown average of of each column in a dataset X using 
# front (coming beforehand) elements + back (coming afterwards) elements.
# The # of front elements should include what's considered the 'current' index 
# (i.e. front = 7 gives average of the current day and the 6 days before it)
def window_average(X, front, back):
    f = X.shape[0]
    lost = front + back
    s = front + back + 1
    c = s
    if len(X.shape) > 1:
        sliders = np.zeros((X.shape[0] - lost, X.shape[1]))
        while c <= f:
            for col in range(X.shape[1]):
                slide = X[c - s: c, col]
                sliders[c - s, col] = np.mean(slide)
            c += 1
    else:
        sliders = np.zeros(X.shape[0] - lost)
        while c <= f:
            slide = X[c - s: c]
            sliders[c - s] = np.mean(slide)
            c += 1
        
    return np.array(sliders)


# Simple L2 norm distance between vectors x and y
def l2(x,y):
    return np.linalg.norm(x-y)


# DTW alignment distance between vectors x and y
# This is mainly a wrapper for easy use of of the dynamic 
# time warping implementation in _dynamic_time_warp.py
# IMPORTANT: Throughout my experiments I use a multiplicative 
# off diagonal penalty of 5. This was chosen for because it 
# matched our intuitive reasoning about how distances between misaligned 
# vectors should present. However, a deeper exploration of parameter values may 
# be worthwile...
def align_distance(x, y, off_diag_penalty = 5):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
        y = y.to_numpy()
        
    # This normalizes so that distance between individual points is always between 0 and 1
    # -- making results more interpretable
    normy = max(x.max(), y.max())
    x = x.copy()
    y = y.copy()
    if normy != 0:
        x /= normy
        y /= normy
    
    # with an alignment penalty, compute the best dtw alignment cost
    off_diag = off_diag_penalty
    mp = [off_diag,off_diag,1]
    ap = [0,0,0]
    cost, align, C= dtw(x, y, distance = l2, mult_penalty = mp, add_penalty = ap)

    # normalize to adjust for the length of the vectors and the size of the multiplicative penalty
    # This just ensures that the distances are on a 0-1 scale
    # and are comparable for time series with different length (again for better interpretability)
    time_norm = off_diag*(len(x) + len(y) - 2) + 1
    return cost/time_norm

        
        
# Given a percentage or level of allowed time overlap 
# between wave segments from a wave pool object (see _wawe_pool.py),
# compute and return an edge list of all edges between wave segments 
# in the time overlap graph
def overlap_graph(percent_overlap, wave_pool_obj):
    edge_list = []
    for i in range(len(wave_pool_obj.key_list) - 1):
        for j in range(i + 1, len(wave_pool_obj.key_list)):
            loc1 = wave_pool_obj.key_list[i]
            loc2 = wave_pool_obj.key_list[j]
            t1 = wave_pool_obj.times[loc1]
            t2 = wave_pool_obj.times[loc2]
            d1 = t1[1] - t1[0]
            d2 = t2[1] - t2[0]
            intersect = min(t1[1], t2[1]) - max(t1[0], t2[0])
            if intersect/d1 >= percent_overlap and intersect/d2 >= percent_overlap:
                edge_list.append((i,j))
    return edge_list


# Given a data matrix, and a list of integer valued indices,
# Remove the rows and columns corresponding to the indices.
def remove_rows_cols(matrix, indices):
    # Indices is a list of integers i where i corresponds to a both a row and a column to drop from the matrix
    indices = sorted(indices, reverse=True)
    for idx in indices:
        # Remove the specified row
        matrix = np.delete(matrix, idx, axis=0)
        # Remove the specified column
        matrix = np.delete(matrix, idx, axis=1)
    return matrix

# Given a data matrix, add a new row and a new column to its end points, i.e. 
# the new column/row is the last column/row in the matrix
def add_row_col(matrix, new_row, new_col):
    # Ensure the new row is a 1D array of the correct length
    new_row = np.array(new_row)
    assert new_row.shape[0] == matrix.shape[1], "New row length must match number of columns in the matrix"
    
    # Append the new row to the matrix
    matrix = np.vstack([matrix, new_row])
    
    # Ensure the new column is a 1D array of the correct length
    new_col = np.array(new_col)
    assert new_col.shape[0] == matrix.shape[0], "New column length must match number of rows in the updated matrix"
    
    # Append the new column to the matrix
    matrix = np.hstack([matrix, new_col.reshape(-1, 1)])
    
    return matrix


# Given two locations loc1 and loc2 
# Both reported as (latitude, longitude) tuples,
# compute and return the distance in miles between them
def haversine(loc1, loc2):
    loc1_rad = [radians(_) for _ in loc1]
    loc2_rad = [radians(_) for _ in loc2]
    result = haversine_distances([loc1_rad, loc2_rad])
    result *= 3958.8  # multiply by Earth radius to get miles
    return result[0,1]



# Compute the disagreement distance between two segmentations of some time series vector x. 
# P,Q are segmentation vectors which take the form P = {p0, p1, ..., pl}
# where each pi is a breakpoint or boundary of the segmentation of x.
# p0 should always be 0 and pl should always be len(x)
def disagreement_distance(P,Q):
    total_disagree = 0
    
    for p in range(len(P) - 1):
        Ep = (P[p+1] - P[p])**2 / 2
        total_disagree += Ep
        
    for q in range(len(Q) - 1):
        Eq = (Q[q+1] - Q[q])**2 / 2
        total_disagree += Eq
        
    U = list(set(P).union(set(Q)))
    U.sort()
    for u in range(len(U) - 1):
        Eu = (U[u+1] - U[u])**2 / 2
        total_disagree -= 2*Eu
        
    return total_disagree
        
    
# Compute the disagreement distance between two segmentation matrices. 
# seg_mat1 and seg_mat2 should be pandas dataframes. They don't 
# necessarily need to have the same shape, but they do need to have 
# the exact same columns. Each column should correspond to a segmentation vector P
# of a time series for a single location. I.e. both dataframes need to have segmentations 
# of the same set of time series vectors. Because segmentation vectors are often varied in length, 
# I often use some maximum number of rows and replace missing entries will NaN values. 
def compute_disagreements(seg_mat1, seg_mat2):
    locations = seg_mat1.columns
    times = seg_mat1.max().max()
    disagreements = np.zeros(len(locations))
    z = math.comb(int(times),2)
    
    for l in range(len(locations)):
        partition1 = seg_mat1.loc[:,locations[l]].to_numpy()
        partition1 = partition1[~np.isnan(partition1)]
        partition2 = seg_mat2.loc[:,locations[l]].to_numpy()
        partition2 = partition2[~np.isnan(partition2)]
        disagreements[l] = disagreement_distance(partition1, partition2)/z
    return disagreements



# Given a distance function which computes distances between wave segments (i.e. align distance or other)
# compute and return a distance matrix D of distances between wave segments from a wpool wave pool object. 
def pairwise_from_pool(wpool, dist):
    n = len(wpool.key_list)
    pairs = list(combinations(wpool.key_list, 2))
    pairs += [(i,i) for i in wpool.key_list]
    
    idx_entries = []
    d_entries = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            idx_entries.append((i,j))
            d_entries.append((wpool.key_list[i],wpool.key_list[j]))
            
    results = []
    for pair in d_entries:
        results.append(dist(pair[0], pair[1]))
    
    row_idx = [i[0] for i in idx_entries]
    col_idx = [i[1] for i in idx_entries]

    # record results in sparse csr matrix
    D = csr_matrix((results, (row_idx, col_idx)), shape = (n,n))
    return D



# from some decreasing curve represented as a finite vector, 
# find an 'elbow' point i.e. the first point where the curves values 
# change less than some given threshold amount
def elbow(curve, threshold, eps = 1.5):
    change_point = -1
    #threshold = np.abs(np.median(np.diff(curve, n= 1)))
    for c in range(len(curve) - 1):
        if change_point == -1:
            #if np.abs(curve[c + 1] - curve[c]) < threshold:
            if curve[c] - curve[c + 1] < threshold:
                change_point = c
            
        #elif np.abs(curve[c + 1] - curve[c]) > threshold:
        elif curve[c] - curve[c + 1] > eps*threshold:
            change_point = -1
            
    return change_point


def elbow2(curve, threshold):
    change_point = None
    for c in range(len(curve) - 1):
        if change_point is None:
            #if np.abs(curve[c + 1] - curve[c]) < threshold:
            if curve[c] - curve[c + 1] < threshold:
                change_point = c
            
        #elif np.abs(curve[c + 1] - curve[c]) > 2*threshold:
        elif curve[c] - curve[c + 1] > 2*threshold:
            change_point = None
            
    return change_point


# Another, more geometric way of findin an elbow point
# draws a line between the beginning an end of the vector 
# and finds the point on the curve with maximum distance to the line
def elbow_distance2(curve):
    x = np.array(range(len(curve)))
    x = x / x[-1]
    curve_shifted = curve - curve[0]
    v = np.array([x[-1], curve_shifted[-1]])
    
    max_i = None
    max_dist = -1
    for i in range(len(x)):
        pi = np.array([x[i], curve_shifted[i]])
        xhat = np.dot(pi, v)/np.dot(v,v)
        proj = xhat * v
        dist = np.linalg.norm(pi - proj)
        
        if dist > max_dist:
            max_i = i
            max_dist = dist
            
    return max_i