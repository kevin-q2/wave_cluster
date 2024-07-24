import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from ._dynamic_time_warp import dtw
from itertools import combinations
from scipy.sparse import csr_matrix


###############################################################################
# The following functions is an assorted set of tools that I use in 
# other parts of the package, as well as in my experiments
#   -Kevin Quinn 1/24
#
###############################################################################


# add gaussian (with chosen mean and std) noise to a data vector
def noise(data, mean = 0, std = 1):
    fuzz = np.random.normal(mean, std, data.shape)
    noisy_D = data + fuzz
    # ensure that the results stay non-negative
    noisy_D = noisy_D.clip(min=0)
    return noisy_D


# take a windowed sum of a dataset with persribed window size
def window(data, window_size):
    rows = data.shape[0]
    columns = data.shape[1]
    windowed_data = np.zeros((rows - window_size + 1, columns))
    for i in range(rows - window_size + 1):
        current_window = data[i:i+window_size,:]
        window_sum = np.sum(current_window,axis=0)
        windowed_data[i,:] = window_sum
        
    return windowed_data


# take a sliding windown average of x using #front (before) elements, current element, and #back (after) elements 
# the # of front elements should include the current index 
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


def window_median(X, front, back):
    f = X.shape[0]
    s = front + back + 1
    c = s
    if len(X.shape) > 1:
        sliders = np.zeros((X.shape[0] - s, X.shape[1]))
        while c < f:
            for col in range(X.shape[1]):
                slide = X[c - s: c, col]
                sliders[c - s, col] = np.median(slide)
            c += 1
    else:
        sliders = np.zeros(X.shape[0] - s)
        while c < f:
            slide = X[c - s: c]
            sliders[c - s] = np.mean(slide)
            c += 1
        
    return np.array(sliders)



def l2(x,y):
    return np.linalg.norm(x-y)

# DTW alignment distance
def align_distance(x, y):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
        y = y.to_numpy()
        
    # normalize so that distance between individual points is always between 0 and 1
    normy = max(x.max(), y.max())
    x = x.copy()
    y = y.copy()
    if normy != 0:
        x /= normy
        y /= normy
    
    # with an alignment penalty, compute the best dtw alignment cost
    off_diag = 5
    mp = [off_diag,off_diag,1]
    ap = [0,0,0]
    cost, align, C= dtw(x, y, distance = l2, mult_penalty = mp, add_penalty = ap)

    # normalize to adjust for the length of the vectors and the size of the multiplicative penalty
    # This just ensures that the distances are on a 0-1 scale
    # and are comparable for time series with different length
    time_norm = off_diag*(len(x) + len(y) - 2) + 1
    return cost/time_norm



# using a vector of cluster labels (n dimensional vector where each entry is 
# a label from [1,..,k]) compute the clustering partition (k dimensional list of lists 
# where each element is assigned to its cluster)
def label_to_cluster(lab):
    clustering = [[] for i in range(int(np.max(lab)) + 1)]
    for l in range(len(lab)):
        clustering[int(lab[l])].append(l)
        
    return clustering


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


def elbow_distance(curve):
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
            
            
# Given two locations loc1 and loc2 
# Both reported as (latitude, longitude) pairs,
# compute and return the distance in miles between them
def haversine(loc1, loc2):
    loc1_rad = [radians(_) for _ in loc1]
    loc2_rad = [radians(_) for _ in loc2]
    result = haversine_distances([loc1_rad, loc2_rad])
    result *= 3958.8  # multiply by Earth radius to get miles
    return result[0,1]
        
        
        


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


def remove_rows_cols(matrix, indices):
    # Indices is a list of integers i where i corresponds to a both a row and a column to drop from the matrix
    indices = sorted(indices, reverse=True)
    for idx in indices:
        # Remove the specified row
        matrix = np.delete(matrix, idx, axis=0)
        # Remove the specified column
        matrix = np.delete(matrix, idx, axis=1)
    return matrix

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


def find_closest_other(Distances, labels, point, graph):
    others = np.unique(labels)
    best = None
    best_distance = np.inf
    for o in others:
        if o != labels[point]:
            labels_o = np.where(labels == o)[0]
            
            overlaps_with = [k for k in labels_o if graph.has_edge(point, k)]
            if len(overlaps_with) >= 1*len(labels_o):
                b_o = np.sum([Distances[point,j] if point < j else Distances[j,point] for j in labels_o])/len(labels_o)
                #print(b_o)
                if b_o < best_distance:
                    best = o
                    best_distance = b_o

    return best_distance


def constrained_silhouette(Distances, labels, graph):
    silhouette_score = 0
    for i in range(len(labels)):
        labels_i = np.where(labels == labels[i])[0]
        if len(labels_i) != 1:
            a_i = np.sum([Distances[i,j] if i < j else Distances[j,i] for j in labels_i if j != i])/(len(labels_i) - 1)
            b_i = find_closest_other(Distances, labels, i, graph)

            if b_i != np.inf:
                silhouette_score += (b_i - a_i)/max(a_i, b_i)
            else:
                silhouette_score += 1
        
    return silhouette_score/len(labels)


def cluster_agreement(labels1, labels2):
    for l in range(int(np.max(labels1))):
        label2_name = None
        for i in range(len(labels1)):
            if labels1[i] == l:
                if label2_name is None:
                    label2_name = labels2[i]
                elif label2_name == labels2[i]:
                    pass
                else:
                    return False
    return True


def ordered_label(labeling):
    label_dict = {}
    new_labeling = np.zeros(len(labeling))
    k = len(np.unique(labeling))
    ki = 0
    for i in range(len(labeling)):
        label_item = labeling[i]
        if label_item not in label_dict.keys() and ki < k:
            label_dict[label_item] = ki
            ki += 1
            
        new_labeling[i] = label_dict[label_item]
        i += 1
    return new_labeling

def ordered_clustering(clustering):
    new_clustering = np.zeros(clustering.shape)
    for i in range(len(clustering)):
        new_clustering[i,:] = ordered_label(clustering[i,:])
    return new_clustering

def unique_clusterings(clusterings):
    labeling_dict = {}
    labeling_ref = np.zeros(len(clusterings))
    current_lab = 0
    for i in range(len(clusterings)):
        labeling = clusterings[i]
        found_equal = False
        for label_key in labeling_dict.keys():
            alt_labeling = labeling_dict[label_key]
            if np.array_equal(labeling, alt_labeling):
                found_equal = True
                labeling_ref[i] = label_key
                
        if not found_equal:
            labeling_dict[current_lab] = labeling
            labeling_ref[i] = current_lab
            current_lab += 1
    return labeling_dict, labeling_ref




# Differences in segmentation methods!
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
        
    

def compute_disagreements(locations, input_space, seg_mat1, seg_mat2):
    disagreements = np.zeros(len(locations))
    z = math.comb(len(input_space),2)
    for l in range(len(locations)):
        partition1 = seg_mat1.loc[:,locations[l]].to_numpy()
        partition1 = partition1[~np.isnan(partition1)]
        partition2 = seg_mat2.loc[:,locations[l]].to_numpy()
        partition2 = partition2[~np.isnan(partition2)]
        disagreements[l] = disagreement_distance(partition1, partition2)/z
    return disagreements



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




# THESE ARE ALSO IN CLUSTER -- you have to figure out what to do with them....
def random_seg_assignment(segs, sizes):
    shuffled = np.random.choice(range(len(segs)), len(segs), replace = False)
    C = []
    i = 0
    for s in sizes:
        C += [list(shuffled[i:i + s])]
        i += s
    return C

def random_seg_assign_sample(segs, sizes, samples):
    random_clusterings = np.zeros((samples, len(segs)))
    for sample in range(samples):
        Clustering = random_seg_assignment(segs, sizes)
        rand_labels = np.zeros(len(segs))
        rand_labels[:] = np.nan
        for clust in range(len(Clustering)):
            for entry in Clustering[clust]:
                rand_labels[entry] = clust
        random_clusterings[sample,:] = rand_labels
    return random_clusterings