import sys
import os
import numpy as np 
import pandas as pd
import networkx as nx
import itertools
sys.path.append("../wave_cluster")
import wave_cluster as wc
import multiprocessing as mp
from multiprocessing import Pool


# import functions to run
sys.path.append("batch/state/scripts/")

##########################################################################################################################################
# Load data:
import subprocess

# Run this external file
sub_result = subprocess.run(["python", "batch/state/scripts/data_analysis_load.py"])

# And import a lot of variables from it which I use below 
from data_analysis_load import *
cpu_count = 16
##########################################################################################################################################


# This script's purpose is to compute clusterings for a range of different parameter values

# This function takes a pair of parameter values (percent overlap, threshold)
# and computes lists of cluster labelings for each segmentation method
def clustering_labels(param_pair):
    p = param_pair[0]
    t = param_pair[1]
    
    uni_edge_list = wc.overlap_graph(p, wpool_uni)
    uni_cliquer = wc.clique_cluster(D_uni, uni_edge_list, linkage = 'complete', threshold = t)
    uni_cliquer.cluster(1)
    uni_analyze = wc.cluster_analyze(uni_cliquer, wpool_uni)
    
    sir_edge_list = wc.overlap_graph(p, wpool_sir)
    sir_cliquer = wc.clique_cluster(D_sir, sir_edge_list, linkage = 'complete', threshold = t)
    sir_cliquer.cluster(1)
    sir_analyze = wc.cluster_analyze(uni_cliquer, wpool_sir)
    
    wav_edge_list = wc.overlap_graph(p, wpool_wav)
    wav_cliquer = wc.clique_cluster(D_wav, wav_edge_list, linkage = 'complete', threshold = t)
    wav_cliquer.cluster(1)
    wav_analyze = wc.cluster_analyze(wav_cliquer, wpool_wav)
    
    return [uni_analyze.labels, sir_analyze.labels, wav_analyze.labels]



# Here I define the range of parameter values to test on 
#overlap_try = np.linspace(0.3,1,15)
#threshold_try = np.linspace(0,0.4,21)
overlap_try = [0.1]
threshold_try = [0.2]
params = list(itertools.product(overlap_try, threshold_try))

# set up empty matrices to store the results
uni_clusterings = np.zeros((len(params), len(wpool_uni.key_list)))
sir_clusterings = np.zeros((len(params), len(wpool_sir.key_list)))
wav_clusterings = np.zeros((len(params), len(wpool_wav.key_list)))

# and then compute the cluster labels for each of the parameter settings in parallel
with Pool(cpu_count) as p:
    results = p.map(clustering_labels, params)
    

# Parse the output results
for r in range(len(results)):
    uni_clusterings[r,:] = results[r][0]
    sir_clusterings[r,:] = results[r][1]
    wav_clusterings[r,:] = results[r][2]
    
    
# And save!
uni_clusterings = pd.DataFrame(uni_clusterings, columns = wpool_uni.key_list)
#uni_clusterings.to_csv('batch/state/data/uni_param_clusterings.csv')

sir_clusterings = pd.DataFrame(sir_clusterings, columns = wpool_sir.key_list)
#sir_clusterings.to_csv('batch/state/data/sir_param_clusterings.csv')

wav_clusterings = pd.DataFrame(wav_clusterings, columns = wpool_wav.key_list)
#wav_clusterings.to_csv('batch/state/data/wav_param_clusterings.csv')
        