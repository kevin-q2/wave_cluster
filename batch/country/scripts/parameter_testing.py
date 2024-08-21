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
sys.path.append("batch/country/scripts/")

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

def get_random_segments():
    max_segments = 10
    segment_size = 60

    rand_cuts = np.zeros((max_segments + 1, len(locations)))

    for l in range(len(locations)):
        loc = locations[l]
        vec = data.loc[:,loc].to_numpy()

        ucuts = unimodal_cuts.loc[:,loc].to_numpy()
        chosen_segs = int((len(ucuts[~np.isnan(ucuts)]) - 1))
        
        rand_seg = wc.random_segment(vec, chosen_segs, segment_size)
        rcuts = rand_seg.generate()

        rand_cuts[:len(rcuts),l] = rcuts
        

    rand_cuts = pd.DataFrame(rand_cuts, columns = locations)
    rand_cuts.replace(0, np.nan, inplace=True)
    rand_cuts.iloc[0,:] = np.zeros(rand_cuts.shape[1])
    return rand_cuts




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
    uni_data = wc.auxiliary_info(uni_cliquer, wpool_uni, [D_uni_miles, D_uni_health], containment_health, data, extra_info = ['uni',None, p, t])
    pair_info += uni_data

    
    # SIR
    sir_edge_list = wc.overlap_graph(p, wpool_sir)
    sir_cliquer = wc.clique_cluster(D_sir, sir_edge_list, linkage = 'complete', threshold = t)
    sir_cliquer.cluster(1)
    sir_data = wc.auxiliary_info(sir_cliquer, wpool_sir, [D_sir_miles, D_sir_health], containment_health, data, extra_info = ['sir',None, p, t])
    pair_info += sir_data
    
    
    # WAV
    wav_edge_list = wc.overlap_graph(p, wpool_wav)
    wav_cliquer = wc.clique_cluster(D_wav, wav_edge_list, linkage = 'complete', threshold = t)
    wav_cliquer.cluster(1)
    wav_data = wc.auxiliary_info(wav_cliquer, wpool_wav, [D_wav_miles, D_wav_health], containment_health, data, extra_info = ['wav',None, p, t])
    pair_info += wav_data

    
    # Random clustering samples
    for s in range(samples_per):
        # random clustering with Unimodal
        r_uni_edge_list = wc.overlap_graph(p, wpool_uni)
        r_uni_cliquer = wc.random_clique_cluster(range(len(wpool_uni.key_list)), r_uni_edge_list)
        r_uni_cliquer.cluster(1)
        r_uni_cliquer.D = D_uni
        r_uni_data = wc.auxiliary_info(r_uni_cliquer, wpool_uni, [D_uni_miles, D_uni_health], containment_health, data, extra_info = ['random_uni',s, p, t])
        pair_info += r_uni_data
        
        # random clustering with SIR
        r_sir_edge_list = wc.overlap_graph(p, wpool_sir)
        r_sir_cliquer = wc.random_clique_cluster(range(len(wpool_sir.key_list)), r_sir_edge_list)
        r_sir_cliquer.cluster(1)
        r_sir_cliquer.D = D_sir
        r_sir_data = wc.auxiliary_info(r_sir_cliquer, wpool_sir, [D_sir_miles, D_sir_health], containment_health, data, extra_info = ['random_sir',s, p, t])
        pair_info += r_sir_data
        
        # random clustering with WAV
        r_wav_edge_list = wc.overlap_graph(p, wpool_wav)
        r_wav_cliquer = wc.random_clique_cluster(range(len(wpool_wav.key_list)), r_wav_edge_list)
        r_wav_cliquer.cluster(1)
        r_wav_cliquer.D = D_wav
        r_wav_data = wc.auxiliary_info(r_wav_cliquer, wpool_wav, [D_wav_miles, D_wav_health], containment_health, data, extra_info = ['random_wav',s, p, t])
        pair_info += r_wav_data
        
        
        # random segment clustering 
        # I have this commented out because its very computationally expensive!!
        # A new distance matrix needs to be computed for every sample!!
        #rand_cuts = get_random_segments()
        #rand_D, rand_wpool = wc.compute_threshold_pairwise(data, rand_cuts, wc.align_distance, percent_overlap = p, cpu_count = cpu_count)
        #rand_edge_list = wc.overlap_graph(p, rand_wpool)
        #rseg_cliquer = wc.clique_cluster(rand_D, rand_edge_list, linkage = 'complete', threshold = t)
        #rseg_cliquer.cluster(1)
        #rseg_data = analyze_clustering(rand_D, rand_wpool, rseg_cliquer.C, [G_miles, G_density], containment_health, data, extra_info = ['random_seg',s, p, t])
        #parameter_info += rseg_data
        
        # random segment random clustering
        rand_cuts = get_random_segments()
        #rand_D, rand_wpool = wc.compute_threshold_pairwise(data, rand_cuts, wc.align_distance, percent_overlap = p, cpu_count = cpu_count)
        rand_wpool = wc.wave_pool(data, rand_cuts, wc.align_distance)
        rand_edge_list = wc.overlap_graph(p, rand_wpool)
        r_cliquer = wc.random_clique_cluster(range(len(rand_wpool.key_list)), rand_edge_list)
        r_cliquer.cluster(1)
        empty_D = np.zeros((len(rand_wpool.key_list), len(rand_wpool.key_list)))
        r_cliquer.D = empty_D
        
        #rand_D_uni_miles = wc.pairwise_from_pool(rand_wpool, miles_dist)
        r_data = wc.auxiliary_info(r_cliquer, rand_wpool, [empty_D, empty_D], containment_health, data, extra_info = ['random',s, p, t])
        pair_info += r_data
    
    return pair_info
    


# Define the parameter values to use!
#overlap_try = np.linspace(0.3,1,15)
#threshold_try = np.linspace(0,0.4,21)
overlap_try = np.linspace(0.5,1,21)
threshold_try = np.linspace(0,0.1,21)

parameter_entries = []
for p in overlap_try:
    for t in threshold_try:
        parameter_entries.append((p,t))

# For every parameter setting, define the number of samples used for random methods:
samples_per = 0

# compute the results for each parameter setting in parallel
with Pool(cpu_count) as p:
    results = p.map(param_info, parameter_entries)
    
# record the results
parameter_info = []
for r in results:
    for entry in r:
        parameter_info.append(entry)   
    
# and save
#columns = ['method', 'sample', 'overlap', 'threshold', 'cluster','cluster_size', 'cost', 'silhouette_score','total_points', 'explained_variance', 'diameter_miles', 'containment_health', 'avg_infections']
columns = ['method', 'sample', 'overlap', 'threshold', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette','health_silhouette', 'geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections']
parameter_info = pd.DataFrame(parameter_info, columns = columns)
parameter_info.to_csv('batch/country/data/parameter_info3.csv')