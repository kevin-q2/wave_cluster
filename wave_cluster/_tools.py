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
    s = front + back + 1
    c = s
    if len(X.shape) > 1:
        sliders = np.zeros((X.shape[0] - s, X.shape[1]))
        while c < f:
            for col in range(X.shape[1]):
                slide = X[c - s: c, col]
                sliders[c - s, col] = np.mean(slide)
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
    diffs = []
    for c in range(len(curve) - 1):
        if np.abs(curve[c + 1] - curve[c]) < threshold:
            return c
        
    return None
            
            
# Given two locations loc1 and loc2 
# Both reported as (latitude, longitude) pairs,
# compute and return the distance in miles between them
def haversine(loc1, loc2):
    loc1_rad = [radians(_) for _ in loc1]
    loc2_rad = [radians(_) for _ in loc2]
    result = haversine_distances([loc1_rad, loc2_rad])
    result *= 3958.8  # multiply by Earth radius to get miles
    return result[0,1]
        