import numpy as np
from sklearn.isotonic import IsotonicRegression
import multiprocessing as mp
from multiprocessing import Pool
from ._dynamic_segment import *
import time


#####################################################################################################################################
#   Fit a unimodal model to any given vector and specific time subsets of it.
#   The unimodal model is made up of an strictly increasing fitted isotonic regression curve, 
#   followed by a strictly decreasing one. The following class allows for local increasing/decreasing 
#   isotonic fitting, which I use as a part of a larger unimodal or k-modal model fitting procedure
#   - Kevin Quinn 12/23
#
#
# Input:
#   data_vec -- input data vector being fit
#
# Attributes:
#   increasing -- boolean variable which decides if fit() and fit_params() will fit an increasing or 
#                   decreasing isotonic regression model to the chosen time period
#
# Methods:
#   fit -- takes as input a time tuple (t1, t2) which are indices to the data vector
#              outputs an error associated with fitting an isotonic regression curve to data_vec[t1:t2] 
#
#   fit_params -- takes as input a time tuple (t1, t2) which are indices to the data vector
#                  outputs an error associated with fitting an isotonic regression curve, along with the fitted curve itself
#
#   generate -- given a list of segmentation cut points which are indices to data_vec, compute multiple fits to 
#           vectors data_vec[cuts[0]:cuts[1]], data_vec[cuts[1]:cuts[2]] , ..., data_vec[cuts[k-1]:cuts[k]]
#           output a list of errors for each fit, a single vector containing all of the fitted curves, and a 
#           list of parameters for each fitted curve.
#
#           **Importantly, in this function I alternate between increasing and decreasing isotonic regression fits 
#           (starting with increasing and then switching to decreasing and continuing until all segments have been fit) 
#
#####################################################################################################################################

class unimodal_fit:
    def __init__(self, data_vec):
        self.data_vec = data_vec
        self.T = len(self.data_vec)
        self.increasing = True
    
    
    def fit_params(self, times):
        t1 = times[0]
        t2 = times[1]
        vec = self.data_vec[t1:t2]
        X = np.array([list(range(t1, t2))])
        X = X.T
        
        iso_reg = IsotonicRegression(increasing = self.increasing).fit(X, vec)
        vec_hat = iso_reg.predict(X)
        
        
        error = np.linalg.norm(vec - vec_hat) #/np.linalg.norm(vec)
        
        # Here I return the full set of information, this is a method I use for more general purposes, but everything else is the same
        return error, vec_hat
    
    
    def fit(self, times):
        error, vec_hat = self.fit_params(times)
        
        # In this function I'm only returning error because I'll use this as part of a larger, parralell process of computing errors 
        # which needs to have simplified function with one output
        return error
    
    
    
    #  From a set of cuts fit a set of unimodal curves to the data!
    def generate(self, cuts):
        self.increasing = True
        errors = []
        fit_vector = np.zeros(len(self.data_vec))
        fit_vector[:] = np.nan
        
        # For each of the time cuts
        for c in range(len(cuts) - 1):
            # alternate between increasing and decreasing fits
            if c % 2 == 0:
                self.increasing = True
            else:
                self.increasing = False
                
            t1 = cuts[c]
            t2 = cuts[c + 1]
            error, subwave = self.fit_params((t1,t2))
            
            errors.append(error)
            fit_vector[t1:t2] = subwave

        return errors, fit_vector

    
    
####################################################################################################################################
# The following function uses the fitting procedures of an unimodal (dual isotonic) regression model to compute a larger 
# matrix of fitting errors, which I use for segmentation
#
# Importantly, this procedure is run in parallel for maximum efficiency
#
# Refer to dynamic_segment.py for more information on how this computed matrix is used 
#
# Input:
#   data_vec -- input data vector being fit
#   segment_size -- (optional) the minimum length that segments must be (if known)
#   cpu_count -- (optional) number of processors to utilize in parallel  
#
# Output:
#   error_table1 -- if T = len(data_vec) then this is a (T + 1) x (T + 1) matrix with each entry (t1,t2) corresponding to 
#                   the error of fitting an INCREASING isotonic regression model to data_vec[t1:t2]
#
#   error_table2 -- if T = len(data_vec) then this is a (T + 1) x (T + 1) matrix with each entry (t1,t2) corresponding to 
#                   the error of fitting a DECREASING isotonic regression model to data_vec[t1:t2]
#
######################################################################################################################################

def compute_kmodal_error_table(data_vec, segment_size = 10, cpu_count = mp.cpu_count() - 1):
    # Initialize the fitting object
    uni = unimodal_fit(data_vec)
    T = len(data_vec)
    
    # errors for increasing segments
    error_table1 = np.zeros((T + 1,T + 1))
    error_table1[:] = np.nan
    
    # errors for decreasing segments
    error_table2 = np.zeros((T + 1,T + 1))
    error_table2[:] = np.nan
    
    # decide which entries of the error table we want to fill in (don't need to compute all of them)
    t_entries = []
    for t1 in range(T):
        for t2 in range(t1 + 1, T + 1):
            if t2 - t1 >= segment_size:
                t_entries.append((t1,t2))


    # for each decided entry, compute the error 
    with Pool(cpu_count) as p:
        results = p.map(uni.fit, t_entries)
        
    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table1[t1,t2] = results[t]
        
        
    uni.increasing=False
    # for each decided entry, compute the error 
    with Pool(cpu_count) as p:
        results = p.map(uni.fit, t_entries)
        
    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table2[t1,t2] = results[t]
    
    return error_table1, error_table2
