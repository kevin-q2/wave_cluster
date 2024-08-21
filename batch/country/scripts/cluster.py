import sys
import numpy as np 
import pandas as pd
import networkx as nx
import time

sys.path.append("../wave_cluster")
import wave_cluster as wc

# import functions to run
sys.path.append("batch/country/scripts/")

##########################################################################################################################################
import subprocess

# If needed do segmentation + pairwise distances beforehand:
#sub_result1 = subprocess.run(["python", "batch/state/scripts/sir_segment.py"])
#sub_result2 = subprocess.run(["python", "batch/state/scripts/pairwise.py"])

# Run this file to load data
sub_result = subprocess.run(["python", "batch/country/scripts/data_analysis_load.py"])

# And import a lot of variables from it which I use below 
from data_analysis_load import *
##########################################################################################################################################


# Specify segmentation setting and Perform clique clustering
wpool = wpool_uni
D = D_uni

overlap = 0.525
threshold = 0.02
edge_list = wc.overlap_graph(overlap, wpool)
cliquer = wc.clique_cluster(D, edge_list, linkage = 'complete', threshold = threshold)
cliquer.cluster(1)

# analyze auxiliary info
cluster_data = wc.auxiliary_info(cliquer, wpool, [D_uni_miles, D_uni_health], containment_health, data)
    
# Convert to pandas and save
cluster_data = pd.DataFrame(cluster_data, columns = ['cluster','cluster_size', 'cost', 'total_points', 'explained_variance',
                                                     'silhouette', 'geo_silhouette', 'health_silhouette','geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
cluster_data.to_csv('batch/country/data/uni_analysis.csv')




# Specify segmentation setting and Perform clique clustering
wpool = wpool_sir
D = D_sir

overlap = 0.5
threshold = 0.03
edge_list = wc.overlap_graph(overlap, wpool)
cliquer = wc.clique_cluster(D, edge_list, linkage = 'complete', threshold = threshold)
cliquer.cluster(1)

# analyze auxiliary info
cluster_data = wc.auxiliary_info(cliquer, wpool, [D_sir_miles, D_sir_health], containment_health, data)
    
# Convert to pandas and save
cluster_data = pd.DataFrame(cluster_data, columns = ['cluster','cluster_size', 'cost', 'total_points', 'explained_variance',
                                                     'silhouette', 'geo_silhouette', 'health_silhouette','geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
cluster_data.to_csv('batch/country/data/sir_analysis.csv')




# Specify segmentation setting and Perform clique clustering
wpool = wpool_wav
D = D_wav

overlap = 0.725
threshold = 0.02
edge_list = wc.overlap_graph(overlap, wpool)
cliquer = wc.clique_cluster(D, edge_list, linkage = 'complete', threshold = threshold)
cliquer.cluster(1)

# analyze auxiliary info
cluster_data = wc.auxiliary_info(cliquer, wpool, [D_wav_miles, D_wav_health], containment_health, data)
    
# Convert to pandas and save
cluster_data = pd.DataFrame(cluster_data, columns = ['cluster','cluster_size', 'cost', 'total_points', 'explained_variance',
                                                     'silhouette', 'geo_silhouette', 'health_silhouette','geo_pairwise', 'health_pairwise', 'containment_health', 'avg_infections'])
cluster_data.to_csv('batch/country/data/wav_analysis.csv')