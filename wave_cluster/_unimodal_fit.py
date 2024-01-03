import numpy as np
from sklearn.isotonic import IsotonicRegression
import multiprocessing as mp
from multiprocessing import Pool
from ._dynamic_segment import *
import time


#####################################################################################################################
#   Fit a unimodal model to any given vector and specific time subsets of it
#   - Kevin Quinn 12/23
#
#
# Input:
#   data_vec -- input data vector being fit
#   population -- population associated with the data
#   initial_window -- initial time window used to compute I0 (as sum of infections over this window)
#   method -- method of computing jacobian (either '2-point' or 'sensitivity')
#
# Methods:
#   sir_fit -- takes as input a time tuple (t1, t2) which are indices to the data vector
#              outputs an error associated with fitting an SIR curve to data_vec[t1:t2] 
#
#   sir_fit_params -- takes as input a time tuple (t1, t2) which are indices to the data vector
#                  outputs an error, a fitted vector, and beta/gamma parameters associated with 
#                   fitting an SIR curve
#
#   generate -- given a list of cuts which are indices to data_vec compute multiple fits to 
#           vectors data_vec[cuts[0]:cuts[1]], data_vec[cuts[1]:cuts[2]] , ..., data_vec[cuts[k-1]:cuts[k]]
#           output a list of errors for each fit, a single vector containing all of the fitted curves, and a 
#           list of parameters for each fitted curve
#
#####################################################################################################################

'''
class unimodal_fit_object:
    def __init__(self, data_vec):
        self.data_vec = data_vec
        self.T = len(self.data_vec)
    
    def unimodal_fit(self, time_cut):
        X1 = np.array([list(range(0, time_cut))])
        X1 = X1.T
        X2 = np.array([list(range(time_cut,self.T))])
        X2 = X2.T
        vec1 = self.data_vec[0:time_cut]
        vec2 = self.data_vec[time_cut:self.T]
        
        # increasing segment
        iso_reg1 = IsotonicRegression(increasing = True).fit(X1, vec1)
        vec_hat1 = iso_reg1.predict(X1)
        # decreasing segment
        iso_reg2 = IsotonicRegression(increasing = False).fit(X2, vec2)
        vec_hat2 = iso_reg2.predict(X2)
        
        
        error1 = np.linalg.norm(vec1 - vec_hat1)
        error2 = np.linalg.norm(vec2 - vec_hat2)
        # In this function I'm only returning error because I'll use this as part of a larger, parralell process of computing errors 
        # which needs to have simplified function with one output
        return (error1, error2)
    
    
    def unimodal_fit_params(self, time_cut):
        X1 = np.array([list(range(0, time_cut))])
        X1 = X1.T
        X2 = np.array([list(range(time_cut,self.T))])
        X2 = X2.T
        vec1 = self.data_vec[0:time_cut]
        vec2 = self.data_vec[time_cut:self.T]
        
        #Increasing Segment
        iso_reg1 = IsotonicRegression(increasing = True).fit(X1, vec1)
        vec_hat1 = iso_reg1.predict(X1)
        # Decreasing segment
        iso_reg2 = IsotonicRegression(increasing = False).fit(X2, vec2)
        vec_hat2 = iso_reg2.predict(X2)
        
        
        error1 = np.linalg.norm(vec1 - vec_hat1)
        error2 = np.linalg.norm(vec2 - vec_hat2)
        
        # Here Instead I return the full set of information, this is a method I use for more general purposes, but everything else is the same
        return error1, error2, vec_hat1, vec_hat2
    
    
    
    #  From a set of cuts fit a set of SIR curves to the data!
    def generate(self, cuts): #, labels = None, type_cuts = None):
        errors = []
        wave = np.zeros(len(self.data_vec))
        wave[:] = np.nan
        
        # TO DO: make this alternating in increase/decrease
        c = cuts[1]
        error1, error2, vec_hat1, vec_hat2 = self.unimodal_fit_params(c)
        errors = [error1,error2]
        wave[:c] = vec_hat1
        wave[c:] = vec_hat2
        return errors, wave,c
'''

class unimodal_fit:
    def __init__(self, data_vec):
        self.data_vec = data_vec
        self.T = len(self.data_vec)
        self.increasing = True
    
    def unimodal_fit(self, times):
        t1 = times[0]
        t2 = times[1]
        vec = self.data_vec[t1:t2]
        X = np.array([list(range(t1, t2))])
        X = X.T
        
        # increasing segment
        iso_reg = IsotonicRegression(increasing = self.increasing).fit(X, vec)
        vec_hat = iso_reg.predict(X)
        
        
        error = np.linalg.norm(vec - vec_hat) #/np.linalg.norm(vec)
        # In this function I'm only returning error because I'll use this as part of a larger, parralell process of computing errors 
        # which needs to have simplified function with one output
        return error
    
    
    def unimodal_fit_params(self, times):
        t1 = times[0]
        t2 = times[1]
        vec = self.data_vec[t1:t2]
        X = np.array([list(range(t1, t2))])
        X = X.T
        
        # increasing segment
        iso_reg = IsotonicRegression(increasing = self.increasing).fit(X, vec)
        vec_hat = iso_reg.predict(X)
        
        
        error = np.linalg.norm(vec - vec_hat) #/np.linalg.norm(vec)
        
        # Here Instead I return the full set of information, this is a method I use for more general purposes, but everything else is the same
        return error, vec_hat
    
    
    
    #  From a set of cuts fit a set of SIR curves to the data!
    def generate(self, cuts): #, labels = None, type_cuts = None):
        errors = []
        wave = np.zeros(len(self.data_vec))
        wave[:] = np.nan
        
        # TO DO: make this alternating in increase/decrease
        for c in range(len(cuts) - 1):
            if c % 2 == 0:
                self.increasing = True
            else:
                self.increasing = False
                
            t1 = cuts[c]
            t2 = cuts[c + 1]
            error, subwave = self.unimodal_fit_params((t1,t2))
            
            errors.append(error)
            wave[t1:t2] = subwave

        return errors, wave

    
    
####################################################################################################################################
# The following function uses the fitting procedures of a unimodal regression model to compute a larger 
# matrix of fitting errors, which I will use in dynamic_segment to segment waves
#
# Importantly, this procedure is run in parallel for maximum efficiency
#
# Refer to dynamic_segment.py for more information on how this computed matrix is used 
#
# Input:
#   data_vec -- input data vector being fit
#   segments -- (optional) the number of wave segments to be used (if known)
#   segment_size -- (optional) the minimum size of segments to be used (if known)
#   cpu_count -- (optional) number of processors to utilize in parallel  
#
# Output:
#   error_table -- if T = len(data_vec) then this is a (T + 1) x (T + 1) matrix with each entry (t1,t2) corresponding to 
#                   the error of fit taken upon data_vec[t1:t2]
#
######################################################################################################################################
    
'''
def compute_unimodal_error_table(data_vec, cpu_count = mp.cpu_count() - 1):
    # Initialize the fitting object
    uni = unimodal_fit_object(data_vec)
    T = len(data_vec)
    error_table = np.zeros((T + 1,T + 1))
    error_table[:] = np.nan
    #parameter_table = np.empty((T + 1, T + 1, 2))
    
    # decide which entries of the error table we want to fill in (don't need to compute all of them)
    t_entries = []
    for t in range(1, T):
        t_entries.append(t)
    

    # for each decided entry, compute the error 
    with Pool(cpu_count) as p:
    #with Pool(10) as p:
        results = p.map(uni.unimodal_fit, t_entries)

    # fill in the error table
    for i in range(len(t_entries)):
        t = t_entries[i]
        error_table[0,t] = results[i][0]
        error_table[t,T] = results[i][1]
        
    
    return error_table



def process_uni_table(data_vec, uni_error_table):
    dp = dynamic_segment(uni_error_table, segments = 2)
    dp.fill_table()
    dp.backtrack()
    uni_errors,uni_wave,uni_cut = unimodal_fit_object(data_vec).generate(dp.cuts)
    return sum(uni_errors), uni_wave, uni_cut




def compute_kmodal_error_table(data_vec, segment_size = 10, cpu_count = mp.cpu_count() - 1):
    # Initialize the fitting object
    uni = unimodal_fit_object(data_vec)
    T = len(data_vec)
    error_table = np.zeros((T + 1,T + 1))
    error_table[:] = np.nan
    #parameter_table = np.empty((T + 1, T + 1, 2))
    
    # decide which entries of the error table we want to fill in (don't need to compute all of them)
    t_entries = []
    for t1 in range(T):
        for t2 in range(t1 + 1, T + 1):
            if t2 - t1 > segment_size:
                t_entries.append((t1,t2))


    # for each decided entry, compute the error 
    #with Pool(cpu_count) as p:
    #with Pool(10) as p:
    #    results = p.map(uni.unimodal_fit, t_entries)
    results = []
    print(len(t_entries))
    for t in t_entries:
        t1 = t[0]
        t2 = t[1]
        start = time.time()
        uni_error_table = compute_unimodal_error_table(data_vec[t1:t2], cpu_count = cpu_count)
        uni_errors, uni_wave, uni_cut = process_uni_table(data_vec[t1:t2], uni_error_table)
        end = time.time()
        print(end - start)
        results.append(uni_errors)
        
        

    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table[t1,t2] = results[t]
    
    return error_table
'''

def compute_kmodal_error_table(data_vec, segment_size = 10, cpu_count = mp.cpu_count() - 1):
    # Initialize the fitting object
    uni = unimodal_fit_object(data_vec)
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
        results = p.map(uni.unimodal_fit, t_entries)
        
    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table1[t1,t2] = results[t]
        
        
    uni.increasing=False
    # for each decided entry, compute the error 
    with Pool(cpu_count) as p:
        results = p.map(uni.unimodal_fit, t_entries)
        
    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table2[t1,t2] = results[t]
    
    return error_table1, error_table2
