import numpy as np
from ._disease_model import *
from ._sir import *
from ._non_linear_fit import *
import multiprocessing as mp
from multiprocessing import Pool


#####################################################################################################################
#   Fit an SIR model to any given vector and specific time subsets of it
#   - Kevin Quinn 11/23
#
#
# Input:
#   data_vec -- input data vector being fit
#   population -- population associated with the data
#   initial_window -- initial time window used to compute I0 (as sum of infections over this window)
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
    
    
class sir_fit:
    def __init__(self, data_vec, population, initialization = 'auto', initial_window = None):
        self.data_vec = data_vec
        self.population = population
        self.initialization = initialization
        self.initial_window = initial_window
    
    
    def fit_params(self, times):
        t1 = times[0]
        t2 = times[1]
        vec = self.data_vec[t1:t2]
        
        # set up the non-linear solver
        if self.initialization == 'auto':
            # NOTE this is my current way to initialize S0, I0, R0 -- could maybe use some work...
            initial_values = np.array([self.population - 10, 10, 0])
            d = len(vec)
            fitting = non_linear_fit(vec, self.population, range(d + 1), initial_values, is_incidence = True, fit_X0 = True)
            
            # this is a really important part!!!
            # I am enforcing beta/gamma to be in the range [0,1]
            # and enforcning S0,I0 to be in the range [0, population]
            bounds = ([0,0,0,0], [1, 1, self.population, self.population])
            theta0 = [0.2,0.1, self.population - 10, 10]
            
        else:
            # if not auto initialization take the sum of the initial window to be
            # the initial I0 value and adjust other parameters around this (Keep R0 = 0)
            new_infect = np.sum(vec[:self.initial_window])
            initial_values = np.array([self.population - new_infect, new_infect, 0])
            d = len(vec[self.initial_window:])
            fitting = non_linear_fit(vec[self.initial_window:], np.array([self.population]), range(d + 1), initial_values, is_incidence = True, fit_X0 = False)   
            bounds = ([0,0], [1, 1])
            theta0 = [0.2,0.1]
        
        
        # make the solver run
        try:
            b_hat, g_hat, X0_hat = fitting.non_linear_solver(bounds, theta0)
            fitting.generator()
            wave_est = fitting.data_est
        except ValueError:
            print('weird error encountered')
            return np.inf, None, None

        
        # calculate error
        if self.initialization == 'auto':
            error = np.linalg.norm(vec - wave_est) #/np.linalg.norm(vec)
        else:
            error = np.linalg.norm(vec[self.initial_window:] - wave_est) #/np.linalg.norm(vec[self.initial_window:])
        
        # Here I return the full set of information, this is a method I use for more general purposes, but everything else is the same
        return error, wave_est, [b_hat, g_hat, X0_hat]
    
    
    
    def fit(self, times):
        error, wave_est, params = self.fit_params(times)
        
        # Here instead I only return the error (So i can use this in parallel processing)
        return error
    
    
    
    #  From a set of cuts fit a set of SIR curves to the data!
    def generate(self, cuts):
        errors = []
        parameters = []
        wave = np.zeros(len(self.data_vec))
        wave[:] = np.nan
        
        for c in range(len(cuts) - 1):
            t1 = cuts[c]
            t2 = cuts[c + 1]
            error, subwave, params = self.fit_params((t1,t2))
            errors.append(error)
            parameters.append(params)
            if self.initialization == 'auto':
                wave[t1:t2] = subwave
            else:
                wave[t1 + self.initial_window:t2] = subwave
            
        return errors, wave, parameters
    
    
    
    
####################################################################################################################################
# The following function uses the fitting procedures in the class above in order to compute a larger 
# matrix of fitting errors, which I often use in another program to segment waves
#
# Importantly, this procedure is run in parallel
#
# Refer to dynamic_segment.py for more information on how this computed matrix is used 
#
# Input:
#   data_vec -- input data vector being fit
#   population -- population associated with the data
#   initial_window -- (optional) initial time window used to compute I0 (as sum of infections over this window)
#   segments -- (optional) the number of wave segments to be used (if known)
#   segment_size -- (optional) the minimum size of segments to be used (if known)
#   cpu_count -- (optional) number of processors to utilize in parallel  
#
# Output:
#   error_table -- if T = len(data_vec) then this is a (T + 1) x (T + 1) matrix with each entry (t1,t2) corresponding to 
#                   the error of fit taken upon data_vec[t1:t2]
#
######################################################################################################################################
    
def compute_sir_error_table(data_vec, population, segment_size = 1, initialization = 'auto', initial_window = 10, cpu_count = mp.cpu_count() - 1):
    # Initialize the fitting object
    sir = sir_fit(data_vec, population, initialization=initialization, initial_window=initial_window)
    T = len(data_vec)
    error_table = np.zeros((T + 1,T + 1))
    error_table[:] = np.nan
    #parameter_table = np.empty((T + 1, T + 1, 2))
    
    # decide which entries of the error table we want to fill in (don't need to compute all of them)
    t_entries = []
    for t1 in range(T):
        for t2 in range(t1 + 1, T + 1):
            if t2 - t1 >= segment_size:
                t_entries.append((t1,t2))
    

    # for each decided entry, compute the error 
    with Pool(cpu_count) as p:
        results = p.map(sir.fit, t_entries)
    p.terminate()
        
    # fill in the error table
    for t in range(len(t_entries)):
        t1 = t_entries[t][0]
        t2 = t_entries[t][1]
        error_table[t1,t2] = results[t]
    
    return error_table