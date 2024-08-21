import sys
import numpy as np 
import pandas as pd
import networkx as nx
import time

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

cpu_count = 4
##########################################################################################################################################


###################################################################
# UNIMODAL:

# Specify setting
wpool = wpool_uni
D = D_uni
overlap = 0.525
edge_list = wc.overlap_graph(overlap, wpool)

# Set up random clusterings
rand_cliquer = wc.random_clique_cluster(D_uni, edge_list)
rand_cliquer.D = D

# Number of random cluster samples to compute
samples = 1000

# For each sample do analysis with auxiliary info
rand_cluster_data = []
for i in range(samples):
    random_clusterings = rand_cliquer.sample(1)
    rc_data = wc.auxiliary_info(rand_cliquer, wpool, [D_uni_miles, D_uni_health], health_index = containment_health, infection_data = data, extra_info = [i])
    rand_cluster_data += rc_data
    
# save results
rand_cluster_data = pd.DataFrame(rand_cluster_data, columns = ['clustering', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette','health_silhouette', 'geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
rand_cluster_data.to_csv('batch/country/data/unimodal_rc_analysis.csv')



###################################################################
# SIR:

# Specify setting
wpool = wpool_sir
D = D_sir
overlap = 0.5
edge_list = wc.overlap_graph(overlap, wpool)

# Set up random clusterings
rand_cliquer = wc.random_clique_cluster(D_sir, edge_list)
rand_cliquer.D = D

# Number of random cluster samples to compute
samples = 1000

# For each sample do analysis with auxiliary info
rand_cluster_data = []
for i in range(samples):
    random_clusterings = rand_cliquer.sample(1)
    rc_data = wc.auxiliary_info(rand_cliquer, wpool, [D_sir_miles, D_sir_health], health_index = containment_health, infection_data = data, extra_info = [i])
    rand_cluster_data += rc_data
    
# save results
rand_cluster_data = pd.DataFrame(rand_cluster_data, columns = ['clustering', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette','health_silhouette','geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
rand_cluster_data.to_csv('batch/country/data/sir_rc_analysis.csv')


###################################################################
# WAV:

# Specify setting
wpool = wpool_wav
D = D_wav
overlap = 0.725
edge_list = wc.overlap_graph(overlap, wpool)

# Set up random clusterings
rand_cliquer = wc.random_clique_cluster(D_wav, edge_list)
rand_cliquer.D = D

# Number of random cluster samples to compute
samples = 1000

# For each sample do analysis with auxiliary info
rand_cluster_data = []
for i in range(samples):
    random_clusterings = rand_cliquer.sample(1)
    rc_data = wc.auxiliary_info(rand_cliquer, wpool, [D_wav_miles, D_wav_health], health_index = containment_health, infection_data = data, extra_info = [i])
    rand_cluster_data += rc_data
    
# save results
rand_cluster_data = pd.DataFrame(rand_cluster_data, columns = ['clustering', 'cluster','cluster_size', 'cost', 'total_points','explained_variance',
            'silhouette', 'geo_silhouette','health_silhouette', 'geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
rand_cluster_data.to_csv('batch/country/data/wav_rc_analysis.csv')