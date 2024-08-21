import sys
import numpy as np 
import pandas as pd
import networkx as nx
import time

sys.path.append("wave_cluster")
import wave_cluster as wc


# import functions to run
sys.path.append("batch/county/scripts/")

##########################################################################################################################################
# Load data:
import subprocess

# Run this external file
sub_result = subprocess.run(["python", "data_analysis_load.py"])
# And import a lot of variables from it which I use below 
from data_analysis_load import *

cpu_count = 8
##########################################################################################################################################


###################################################################
# UNIMODAL:

# Specify setting
wpool = wpool_uni
D = D_uni
overlap = 0.825
edge_list = wc.overlap_graph(overlap, wpool)


#rand_cliquer = wc.random_clique_cluster(range(len(wpool.key_list)), edge_list)
rand_cliquer = wc.random_clique_cluster(D, edge_list)

# Number of random cluster samples to compute
samples = 1000

# For each sample do analysis with auxiliary info
rand_cluster_data = []
for i in range(samples):
    random_clusterings = rand_cliquer.sample(1)
    rc_data = wc.auxiliary_info(rand_cliquer, wpool, [D_uni_miles], health_index = None, infection_data = data, extra_info = [i])
    rand_cluster_data += rc_data

rand_cluster_data = pd.DataFrame(rand_cluster_data, columns = ['sample','cluster','cluster_size', 'cost', 'total_points', 'explained_variance', 
                                                    'silhouette', 'geo_silhouette', 'geo_pairwise', 'containment_health', 'avg_infections'])

rand_cluster_data.to_csv('batch/county/data/northeast_random_cluster.csv')