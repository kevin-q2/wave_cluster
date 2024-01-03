import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from .sir_fits import *


##########################################################################################################################
# Dynamic Program for splitting a data vector into segments
#
#   INPUT:
#       error_table -- square matrix representing the errors computed on different segments of some data
#                   for example, suppose we have some data vector data_vec such that T = len(data_vec)
#                   Then error table should be a (T + 1) x (T + 1) matrix where each entry (t1,t2) corresponds to some 
#                   chosen measure of error upon segment data_vec[t1:t2]
#
#       segments -- the number of distinct segments to split the data into
#       segment_Size -- the minimum length a segment can take
#  
#   ATTRIBUTES:
#       segment_table -- a (segments x (T + 1)) dimensional table where each entry ij corresponds to 
#                        the error produced by splitting the first j data points into i segments
#
#       tracking_table -- a (segments x (T + 1)) dimensional table where each entry ij keeps track of which entry k in the row i-1
#                        was used to form the error for ij.  (dynamic formula)
#
#       cuts -- a list of the optimal set of cuts used to segment the data
#
#  METHODS:
#       fill_table() -- takes the error table input and fills in segment_table and tracking_table
#       backtrack() -- once segment_table and tracking_table have been computed, use them to compute the 
#                       optimal list of cuts
#
#
# ---Kevin Quinn 11/23
#############################################################################################################################


class dynamic_segment:
    def __init__(self, error_table, segments, segment_size = 1, error_table2 = None):
        self.segments = segments
        self.segment_size = segment_size
        self.T = len(error_table) - 1
        
        if self.T < self.segments * self.segment_size:
            raise ValueError("segments required will be longer than data")

        self.error_table = error_table    
        if error_table2 is not None:
            self.error_table1 = error_table
            self.error_table2 = error_table2
        else:
            self.error_table2 = None
        
        self.segment_table = np.zeros((self.segments, self.T + 1))
        self.tracking_table = np.zeros((self.segments, self.T + 1))
        
        self.cuts = None
        
           
            
    
    # fills the segment and tracking tables
    def fill_table(self):
        for k in range(self.segments):
            if self.error_table2 is not None:
                if k % 2 == 0:
                    self.error_table = self.error_table1
                else:
                    self.error_table = self.error_table2
                
            
            # make room for segments that came before 
            start = k * self.segment_size
            #start = 0
            # make some more room for forthcoming segments
            #end = self.T - (self.segments-(k+1))*self.segment_size
            end = self.T
            
            # the first segment must start at 0
            if k == 0:
                for t2 in range(self.segment_size, end + 1):
                    s_fit = self.error_table[0,t2]
                    if np.isnan(s_fit):
                            print((0,t2))
                    self.segment_table[k,t2] = s_fit
            
            # and the last segment must end at the last time step
            elif k == self.segments - 1:
                min_entry = None
                min_error = np.inf
                
                for t1 in range(start,self.T - self.segment_size + 1):
                        new_error = self.error_table[t1,self.T]
                        segment_errors = self.segment_table[k - 1, t1] + new_error
                        
                        if segment_errors < min_error:
                            min_entry = t1
                            min_error = segment_errors
                            
                    
                self.segment_table[k,self.T] = min_error
                self.tracking_table[k,self.T] = min_entry
                
            # Everything else in between is subject to iteration over multiple time starts/ends
            else:
                for t2 in range(start + self.segment_size, end+1):
                    min_entry = None
                    min_error = np.inf

                    for t1 in range(start,t2 - self.segment_size + 1):
                        new_error = self.error_table[t1,t2]
                        segment_errors = self.segment_table[k - 1, t1] + new_error
                        
                        if segment_errors < min_error:
                            min_entry = t1
                            min_error = segment_errors
                            
                    
                    self.segment_table[k,t2] = min_error
                    self.tracking_table[k,t2] = min_entry
                    
                            
    # parses the segment and tracking tables to find the optimal cuts                    
    def backtrack(self, num_segments = None):
        cuts = [self.T]
        new_cut = cuts[0]
        if num_segments is None:
            k = self.segments - 1
        else:
            k = num_segments - 1
        
        while k >= 0:
            c = int(self.tracking_table[k, new_cut])

            new_cut = c
            cuts = [new_cut] + cuts
            k -= 1
        
        self.cuts = np.array(cuts)
        #return self.cuts
        
        
    def tester(self, waves, error_func):
        errs = []
        
        for w in waves:
            self.backtrack(w)
            error = error_func(self.cuts)
            errs.append(error)
            
        return errs
