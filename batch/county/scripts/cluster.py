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
import subprocess

# If needed do segmentation + pairwise distances beforehand:
#sub_result1 = subprocess.run(["python", "batch/county/scripts/sir_segment.py"])
#sub_result2 = subprocess.run(["python", "batch/county/scripts/pairwise.py"])

# Run this file to load data
sub_result = subprocess.run(["python", "batch/county/scripts/data_analysis_load.py"])

# And import a lot of variables from it which I use below 
from data_analysis_load import *
##########################################################################################################################################


# Specify segmentation setting and Perform clique clustering
wpool = wpool_uni
D = D_uni

# Choose parameters 
overlap = 0.950
threshold = 0.010
edge_list = wc.overlap_graph(overlap, wpool)
cliquer = wc.clique_cluster(D, edge_list, linkage = 'complete', threshold = threshold)
cliquer.cluster(1)

# Save results if necessary!
saveC = pd.DataFrame(cliquer.C)
saveC.to_csv('batch/county/data/east_uni_clusters.csv')

# OR load results if they are pre-computed.
'''
clusters = pd.read_csv('batch/county/data/northeast_county_clusters.csv', index_col = 0)
C = []
for i in range(len(clusters)):
    cluster_loc = clusters.iloc[i,:].to_list()
    c = [i for i in cluster_loc if not np.isnan(i)]    
    c = [int(i) for i in c]
    C.append(c)

cliquer.C = C
'''


# analyze auxiliary info
cluster_data = wc.auxiliary_info(cliquer, wpool, [D_uni_miles], health_index = None, infection_data = data)
    
# Convert to pandas and save
cluster_data = pd.DataFrame(cluster_data, columns = ['cluster','cluster_size', 'cost', 'total_points', 'explained_variance',
                                                     'silhouette', 'geo_silhouette', 'geo_pairwise', 'containment_health', 'avg_infections'])
cluster_data.to_csv('batch/county/data/east_unimodal_analysis.csv')