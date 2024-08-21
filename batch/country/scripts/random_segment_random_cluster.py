import sys
import os
import numpy as np
import pandas as pd
import time
import networkx as nx
from itertools import combinations

sys.path.append("wave_cluster")
import wave_cluster as wc


# import functions to run
sys.path.append("batch/country/scripts/")

##########################################################################################################################################
# Load data:
import subprocess

# Run this external file
sub_result = subprocess.run(["python", "data_analysis_load.py"])
# And import a lot of variables from it which I use below 
from data_analysis_load import *

cpu_count = 16
#########################################################################################################################


# Get a full table of random segments
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


# specify setting 
overlap = 0.525
threshold = 0.02
#dist_graph = G_miles

# Specify total number of random segmentation samples
num_segment_samples = 1000

# And for each random segmentation how many random clustering samples to do
num_cluster_samples = 1


# for a single sample, do random segmentation and random clustering
def get_random_segment_random_cluster(sample):
    # random segmentation and DTW computations
    rand_cuts = get_random_segments()
    rand_D, rand_wpool = wc.compute_threshold_pairwise(data, rand_cuts, wc.align_distance, percent_overlap = overlap, cpu_count = cpu_count)
    rand_edge_list = wc.overlap_graph(overlap, rand_wpool)
    # Random Clustering
    rand_cliquer = wc.random_clique_cluster(rand_D, rand_edge_list)
    rand_cliquer.D = rand_D
    D_rand_miles = wc.pairwise_from_pool(rand_wpool, miles_dist)
    D_rand_health, rand_health_wpool = wc.compute_threshold_pairwise(containment_health, rand_cuts, wc.align_distance, percent_overlap = overlap, cpu_count = cpu_count)
    
    # Alternatively --without actual distance computation
    #rand_cuts = get_random_segments()
    #rand_wpool = wc.wave_pool(data, rand_cuts, wc.align_distance)
    #rand_edge_list = wc.overlap_graph(overlap, rand_wpool)
    #rand_cliquer = wc.random_clique_cluster(range(len(rand_wpool.key_list)), rand_edge_list)
    #empty_D = np.zeros((len(rand_wpool.key_list), len(rand_wpool.key_list)))
    #rand_cliquer.D = empty_D

            
    # Parse random clusterings
    rand_info = []
    for i in range(num_cluster_samples):
        random_clusterings = rand_cliquer.sample(1)
        rand_info += wc.auxiliary_info(rand_cliquer, rand_wpool, [D_rand_miles, D_rand_health], health_index = containment_health, infection_data = data, extra_info = [sample])
        
    return rand_info
            

# I use this as a way of splitting the task up into different jobs 
# (i.e. I usually run these remotely on a compute cluster and split up the tasks since they can be intensive)
job_id = os.getenv('SGE_TASK_ID')
sample_num = (int(job_id) - 1)*num_segment_samples
random_results = []
for s in range(num_segment_samples):
    rand_res = get_random_segment_random_cluster(sample_num + s)
    random_results += rand_res
                
# save results                
#rand_segment_rand_cluster_data = pd.DataFrame(random_results, columns = ['sample','cluster','cluster_size', 'cost', 'silhouette_score','total_points','explained_variance', 'diameter_miles', 'containment_health', 'avg_infections'])
rand_segment_rand_cluster_data = pd.DataFrame(random_results, columns = ['sample', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette','health_silhouette', 'geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
rand_segment_rand_cluster_data.to_csv('batch/country/data/random_segment_random_clusters_' + str(job_id) + '.csv')