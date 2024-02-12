import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians


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
def elbow(curve, threshold):
    change_point = None
    #threshold = np.abs(np.median(np.diff(curve, n= 1)))
    for c in range(len(curve) - 1):
        if change_point is None:
            #if np.abs(curve[c + 1] - curve[c]) < threshold:
            if curve[c] - curve[c + 1] < threshold:
                change_point = c
            
        #elif np.abs(curve[c + 1] - curve[c]) > threshold:
        elif curve[c] - curve[c + 1] > threshold:
            change_point = None
            
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