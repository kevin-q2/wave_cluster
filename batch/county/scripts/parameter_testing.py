import sys
import os
import numpy as np 
import pandas as pd
import networkx as nx
sys.path.append("wave_cluster")
import wave_cluster as wc
import multiprocessing as mp
from multiprocessing import Pool


# import functions to run
sys.path.append("batch/county/scripts/")

##########################################################################################################################################
# Load data:
import subprocess

# Run this external file
sub_result = subprocess.run(["python", "data_analysis_load.py"])

# And import a lot of variables from it which I use below 
from data_analysis_load import *
#cpu_count = int(os.getenv('NSLOTS'))
cpu_count = 16
##########################################################################################################################################


# In this script I implement the collection of auxiliary analysis info for clusterings 
# over a range of parameter settings, and with all of our clustering methods (all segmentation variations and also including random methods)


# This function takes a parameter pair (percent ovelrap, threshold) and 
# computes the analysis results for every clustering method
def param_info(param_pair):
    p = param_pair[0]
    t = param_pair[1]
    
    pair_info = []
    # unimodal
    uni_edge_list = wc.overlap_graph(p, wpool_uni)
    uni_cliquer = wc.clique_cluster(D_uni, uni_edge_list, linkage = 'complete', threshold = t)
    uni_cliquer.cluster(1)
    uni_data = wc.auxiliary_info(uni_cliquer, wpool_uni, [D_uni_miles], health_index = None, infection_data = data, extra_info = ['uni',None, p, t])
    pair_info += uni_data
    
    return pair_info
    


# Define the parameter values to use!
overlap_try = np.linspace(0.5,1,21)
threshold_try = np.linspace(0,0.1,21)
parameter_entries = []
for p in overlap_try:
    for t in threshold_try:
        parameter_entries.append((p,t))


# compute the results for each parameter setting in parallel
with Pool(cpu_count) as p:
    results = p.map(param_info, parameter_entries)
    
# record the results
parameter_info = []
for r in results:
    for entry in r:
        parameter_info.append(entry) 

columns = ['method', 'sample', 'overlap', 'threshold', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette', 'geo_pairwise', 'containment_health', 'avg_infections']
parameter_info = pd.DataFrame(parameter_info, columns = columns)
parameter_info.to_csv('batch/county/data/northeast_parameter_info.csv')

