import numpy as np
import scipy.sparse
import networkx as nx
import itertools 
import time
import math
import random
import copy
from ._tools import remove_rows_cols,add_row_col 

#####################################################################################################################
#   Clustering with Cliques -- Given a matrix of distances and a list of edges,
#   cluster/partition items so that every cluster is a clique (any pair of items 
#   in the cluster is connected by an edge) and within cluster distances are minimized.
#   The driver behind this is a custom hierarchical clustering algorithm. At every iteration 
#   the algorithm joins two clusters as long as the joined group still forms a clique.
#   
#   INPUT:
#       D -- (n x n) Distance matrix between items/nodes

#       edge_list -- A list of edges in tuple form, such as (i,j), indicating that the items/nodes 
#                   associated with the indices i,j of in distance matrix D are adjacent. 
#                   Importantly, a,b must be of type int to properly index D. 

#       linkage -- Method for joining clusters at every iteration. Currently I only 
#               support 'single' linkage or 'complete' linkage

#       threshold -- Value at which to stop joining clusters. If no two joinable clusters with a 
#                   linkage distance > threshold may be found, then terminate the algorithm
#       
#   ATTRIBUTES:
#       G -- NetworkX graph formed from the input edge list 
#
#       C -- List of clusters, where each cluster is reported as a list of item indices present in the 
#            cluster
#       
#       D_cluster -- a distance matrix updated at each iteration to keep track of distances between clusters. 
#                   Importantly, I'll maintain that zero valued entries of D_cluster indicate that the clusters 
#                   cannot be joined to form a clique. This also implies that you'll need non-zero distances for 
#                   unique points (i.e. statisfies metric properties). Doing this allows for an efficient 
#                   way to keep track of which clusters may be joined together. 
#
#       last_links -- keeps track of the two items which were most recently joined into a single cluster
#
#       Linkage matrix? Either need to fully implement or delete this
#
#   METHODS:
#       cluster(k) -- given a input number of clusters, k, run the hierarchical clustering algorithm 
#                   on the inputs above until you've either found exactly k clusters, or otherwise 
#                   the algorithm is stopped early because of threshold conditions or because its no 
#                   longer possible to join any more clusters!
# 
#       The rest of the methods are described individually below       
#
#
#   Kevin Quinn 11/23
#####################################################################################################################


class clique_cluster:
    def __init__(self, D, edge_list, linkage = 'complete', threshold = np.inf):
        self.D = D
        self.G = nx.from_edgelist(edge_list)
            
        self.C = [[i] for i in range(D.shape[1])]
        self.D_cluster = D.copy()
        
        if linkage == 'single':
            self.linkage = self.single_linkage
        elif linkage == 'complete':
            self.linkage = self.complete_linkage
            
        self.threshold = threshold
        
        self.last_links = None
        
        #self.linkage_matrix = []
        #self.linkage_index = len(self.C) - 1
        #self.linkage_names = [i for i in range(D.shape[1])]
        
        self.distance_preprocess()
    
        
    def distance_preprocess(self):
        # Pre-processes (A COPY) of the entries of the distance matrix so that distances between 
        # items i,j are only non-zero if there exists an edge between them
        #
        # NOTE: For now I am using 0 entries to represent an absence of edges between clusters 
        #       This doesn't account for the fact that distances could maybe be 0 between clusters...
        #       I'm going to assume for now that that will never happen...

        self.D_cluster = self.D_cluster.toarray()
        D_mask = np.zeros(self.D_cluster.shape)
                
        for (i,j) in list(self.G.edges):
            D_mask[min(i,j),max(i,j)] = 1
            
        self.D_cluster = scipy.sparse.csr_matrix(self.D_cluster * D_mask)
                    
                    
    def check_clique(self, c_i, c_j):
        # Given indices c_i and c_j, check if joining clusters c_i and c_j
        # would form a larger clique by assessing whether or not their 
        # entry in D_cluster is non-zero
        
        #if self.D_cluster[c_i, c_j] == np.inf:
        if self.D_cluster[min(c_i, c_j), max(c_i, c_j)] == 0:
            return False
        else:
            return True


    def single_linkage(self, idx):
        # compute the single linkage distance between a cluster indexed by idx and the newly 
        # formed cluster which is a combination of items in last_links
        dist1 = self.D_cluster[min(idx, self.last_links[0]), max(idx, self.last_links[0])]
        dist2 = self.D_cluster[min(idx, self.last_links[1]), max(idx, self.last_links[1])]
        if dist1 == 0 or dist2 == 0:
            return 0
        else:
            return min(dist1, dist2)
    

    def complete_linkage(self, idx):
        # compute the complete linkage distance between between a cluster indexed by idx and the newly 
        # formed cluster which is a combination of items in last_links
        dist1 = self.D_cluster[min(idx, self.last_links[0]), max(idx, self.last_links[0])]
        dist2 = self.D_cluster[min(idx, self.last_links[1]), max(idx, self.last_links[1])]
        if dist1 == 0 or dist2 == 0:
            return 0
        else:
            return max(dist1, dist2)
    
    
    def distance_update(self):
        # Update the distances of D_cluster
        # Specifically, this is designed to be called after joining two clusters.
        # It will remove the rows/columns associated with the joined cluster, 
        # and add a new row/column with computed distances for the new cluster
        # onto the end of the matrix. 
        
        Dc = self.D_cluster.toarray()
        Dc = remove_rows_cols(Dc, self.last_links)
        
        new_dists = []
        for i in range(len(self.C) - 1):
            if i >= self.last_links[1] - 1:
                diff_i = i + 2
            elif i >= self.last_links[0]:
                diff_i = i + 1
            else:
                diff_i = i
                
            new_dists.append(self.linkage(diff_i))
            
        new_row = np.zeros(len(new_dists))
        new_col = np.array(new_dists + [0])
        Dc = add_row_col(Dc, new_row, new_col)
        self.D_cluster = scipy.sparse.csr_matrix(Dc)
            
    
    def cluster_update(self, c_i, c_j, v_ij):
        # Given indices of clusters c_i and c_j and the distance value v_ij between them,
        # update cluster list C by joining clusters c_i and c_j. Specifically, this removes 
        # clusters c_i and c_j, and replaces them with the single cluster formed by the 
        # union of their items. 
        
        new_cluster = [self.C[c_i] + self.C[c_j]]
        self.C = [self.C[i] for i in range(len(self.C)) if i != c_i and i != c_j] + new_cluster
        self.last_links = [c_i, c_j]
        
        #self.linkage_index += 1
        #self.linkage_matrix.append([self.linkage_names[c_i], self.linkage_names[c_j], v_ij])
        #self.linkage_names = [self.linkage_names[i] for i in range(len(self.linkage_names)) if i != c_i and i != c_j] + [self.linkage_index]
    
    
    def cluster(self, k):
        while len(self.C) > k:
            r,c,v = scipy.sparse.find(self.D_cluster)
            
            # We want to argsort the values of D breaking ties randomly
            # Create an array of random numbers with the same shape as v
            random_tiebreakers = np.random.rand(v.shape[0])

            # Combine the original array and random tiebreakers into a structured array
            structured_array = np.core.records.fromarrays([v, random_tiebreakers], 
                                                        names='values,rand')

            # Use np.argsort on the structured array to get the sorted indices
            v_sort = np.argsort(structured_array, order=['values', 'rand'])
            
            # Now greedily find the smallest cost adjointment of clusters 
            # which is able to form a clique
            start = 0
            boo = False
            while not boo and start < len(v):
                min_v = v_sort[start]
                min_r = r[min_v]
                min_c = c[min_v]
                boo = self.check_clique(min_r, min_c)
                
                start += 1
            
            # No more cliques or distance threshold reached!
            if boo == False:
                return
            
            elif start < len(v):
                if v[v_sort[start]] > self.threshold:
                    return
            
            self.cluster_update(min_r, min_c, v[min_v])
            self.distance_update()
    


###############################################################################################################################
# Randomly cluster a set of nodes/items with the only constraint being that each cluster 
# must be a clique (i.e. every pair of items in the cluster is adjacent according to an input edge list)
#
#   INPUT:
#       D -- (n x n) Distance matrix with REAL distances between items/nodes

#       edge_list -- list of tuples (i,j) describing adjacency between the nodes in node_list
#
#   ATTRIBUTES:
#       C -- List of clusters, where each cluster is reported as a list of item indices present in the cluster
#
#       G -- NetworkX graph formed from the input edge list 
#
#   METHODS:
#       sample(k) -- compute a new random k clustering using a dummy distance matrix 
#
##############################################################################################################################



class random_clique_cluster(clique_cluster):
    def __init__(self, D, edge_list):
        self.D = D
        self.G = nx.from_edgelist(edge_list)

        self.C = [[i] for i in range(D.shape[1])]
        self.C_copy = [[i] for i in range(D.shape[1])]
        
        # Dummy distance matrix -- every distance between clusters == 1
        I = np.ones(D.shape)
        np.fill_diagonal(I,0)
        self.D_cluster = scipy.sparse.csr_matrix(I)
        
        # set to complete linkage, but since we want to join clusters randomly, 
        # we'll set the distances so that this doesn't really matter
        self.linkage = self.complete_linkage
        self.threshold = np.inf
        self.last_links = None
        
        # Pre-process so that cliques are easier to work with
        # sets distances to 0 if no edge exists between items
        self.distance_preprocess()
        
        self.D_cluster_copy = self.D_cluster.copy()
        
        
        
    def sample(self, k):
        # reset clustering and distances:
        self.C = copy.deepcopy(self.C_copy)
        self.D_cluster = self.D_cluster_copy.copy()
        self.last_links = None
        
        # and cluster:
        self.cluster(k)
        
        
'''
            
class random_clique_cluster(clique_cluster):
    def __init__(self, node_list, edge_list):
        self.edge_list = edge_list
        self.node_list = node_list
        self.G = nx.from_edgelist(edge_list)
        self.C = [[i] for i in range(len(node_list))]
        self.last_links = None
        
        # set to complete linkage, but since we want to join clusters randomly, 
        # we'll set the distances so that this doesn't really matter
        self.linkage = self.complete_linkage
        
        # Dummy distance matrix -- every distance between clusters == 1
        self.D = np.zeros((len(node_list), len(node_list))) + 1
        
        # Pre-process so that cliques are easier to work with
        # sets distances to 0 if no edge exists between items
        self.distance_preprocess()
        
        
               
    def cluster_update(self, c_i, c_j, v_ij):
        # Given indices of clusters c_i and c_j and the distance value v_ij between them,
        # update cluster list C by joining clusters c_i and c_j. Specifically, this removes 
        # clusters c_i and c_j, and replaces them with the single cluster formed by the 
        # union of their items. 
        new_cluster = [self.C[c_i] + self.C[c_j]]
        self.C = [self.C[i] for i in range(len(self.C)) if i != c_i and i != c_j] + new_cluster
        self.last_links = [c_i, c_j]
        

           
    def cluster(self,k):
        while len(self.C) > k:
            # sample randomly from pairs of cliques
            stop = math.comb(len(self.C),2)
            boo = False
            start = 0
            while not boo and start < stop:
                rand_comb = random.sample(range(len(self.C)), 2)
                rand_r = rand_comb[0]
                rand_c = rand_comb[1]
                boo = self.check_clique(rand_r, rand_c)
                start += 1
                
                
            if boo == False:
                # If we couldn't find a clique, go through every possible combination 
                # deterministically to make sure there really is nothing left
                randlist = list(itertools.combinations(range(len(self.C)), 2))
                order = np.random.choice(range(len(randlist)), len(randlist))
                start2 = 0
                while not boo and start2 < len(order):
                    rand_r = randlist[order[start2]][0]
                    rand_c = randlist[order[start2]][1]
                    boo = self.check_clique(rand_r, rand_c)
                    start2 += 1
                    
                if boo == False:
                    return
            
            self.cluster_update(rand_r, rand_c, 0)
            self.distance_update()
            
    
    def sample(self, k, num_samples):
        random_clusterings = np.zeros((num_samples, len(self.node_list)))
        for s in range(num_samples):
            self.C = [[i] for i in range(len(self.node_list))]
            self.cluster(k)
            rand_labels = np.zeros(len(self.node_list))
            rand_labels[:] = np.nan
            for clust in range(len(self.C)):
                for i in self.C[clust]:
                    rand_labels[i] = clust
            random_clusterings[s,:] = rand_labels
            
        return random_clusterings
            
            


def random_seg_assignment(segs, sizes):
    shuffled = np.random.choice(range(len(segs)), len(segs), replace = False)
    C = []
    i = 0
    for s in sizes:
        C += [list(shuffled[i:i + s])]
        i += s
    return C

def random_seg_assign_sample(segs, sizes, samples):
    random_clusterings = np.zeros((samples, len(segs)))
    for sample in range(samples):
        Clustering = random_seg_assignment(segs, sizes)
        rand_labels = np.zeros(len(segs))
        rand_labels[:] = np.nan
        for clust in range(len(Clustering)):
            for entry in Clustering[clust]:
                rand_labels[entry] = clust
        random_clusterings[sample,:] = rand_labels
    return random_clusterings
'''    
    
    
    
            
##############################################################################################################
#   A framework for analyzing a clustering output by either clique_cluster or random_cluster
#   that I rely heavily on throughout my experiments. 
#
#   INPUT:
#       cluster_object -- clique_cluster or random_cluster class object with computed clusters
#
#       wave_pool object -- Contains information about the infection waves which are treated as
#                          nodes/items in the clustering process. For more information, see 
#                           wave_pool.py to understand what's contained in this class
#
#   ATTRIBUTES:
#       clustering -- the list of clusters associated with the input cluster object
#
#       labels -- a list of labels assigning each item in the wave_pool to a cluster
#
#   METHODS:
#       desribed individually below
##############################################################################################################
            
            
class cluster_analyze:
    def __init__(self, cluster_object, segment_pool):
        self.cluster_object = cluster_object
        self.segment_pool = segment_pool
        self.D = self.cluster_object.D
        self.G = self.cluster_object.G
        self.clustering = self.cluster_object.C
        self.labels = self.cluster_to_label(self.clustering)
        

            
    def cluster_to_label(self, clustering):
        # takes an input clustering (list of lists) and returns a list of labels 
        # assigning each item a numberical value according to the index of their cluster in clustering
        lens = [len(i) for i in clustering]
        labels = np.zeros(np.sum(lens))
        
        for c in range(len(clustering)):
            for k in clustering[c]:
                labels[k] = c
                
        return labels
        

    def label_to_cluster(self, labels):
        # In a reverse role, this takes a list of labels corresponding to the indices of each items 
        # cluster, and outputs a clustering (list of lists)
        clustering = [[] for i in range(int(np.max(labels)) + 1)]
        for l in range(len(labels)):
            clustering[int(labels[l])].append(l)
            
        return clustering
    
    def avg_pairwise_distances(self, Distances, cluster):
        # for a given distance matrix and a given cluster, 
        # compute the average of pairwise distances between elements of the cluster
        if len(self.clustering[cluster]) != 1:
            clust_dists = []
            clust_pairs = list(itertools.combinations(self.clustering[cluster], 2))
            
            for p in clust_pairs:
                if p[0] < p[1]:
                    clust_dists.append(Distances[p[0],p[1]])
                else:
                    clust_dists.append(Distances[p[1],p[0]])

            return np.mean(clust_dists)
        else:
            return 0
    
    def avg_cost(self, cluster):
         # this is just a wrapper for avg_pairwise_distance that allows me to 
        # compute using the initialized distances self.D
        return self.avg_pairwise_distances(self.D, cluster)
        
    
    def complete_linkage_cost(self, cluster):
        # for a given cluster, find the largest distance between any two elements
        if len(cluster) != 1:
            clust_costs = []
            for k in range(len(cluster)):
                for k2 in range(k + 1, len(cluster)):
                    if cluster[k] < cluster[k2]:
                        cost = self.D[cluster[k],cluster[k2]]
                    else:
                        cost = self.D[cluster[k2],cluster[k]]
                        
                    clust_costs.append(cost)

            return np.max(clust_costs)
        else:
            return 0
        
        
    def single_linkage_cost(self, cluster):
        # for a given cluster, find the smallest distance between any two elements
        if len(cluster) != 1:
            clust_costs = []
            for k in range(len(cluster)):
                for k2 in range(k + 1, len(cluster)):
                    if cluster[k] < cluster[k2]:
                        cost = self.D[cluster[k],cluster[k2]]
                    else:
                        cost = self.D[cluster[k2],cluster[k]]
                        
                    clust_costs.append(cost)

            return np.min(clust_costs)
        else:
            return 0
        
        
    def find_closest_other(self, Distances, point):
        # given Distances and a single point/item, find the closest other cluster
        # (a cluster to which point does not already belong to) by searching for 
        # alternative clusters that have the smallest sum of distances between all 
        # of their items and the input point
        
        others = np.unique(self.labels)
        best = None
        best_distance = np.inf
        for o in others:
            if o != self.labels[point]:
                labels_o = np.where(self.labels == o)[0]
                
                overlaps_with = [k for k in labels_o if self.G.has_edge(point, k)]
                if len(overlaps_with) >= 1*len(labels_o):
                    b_o = np.sum([Distances[point,j] if point < j else Distances[j,point] for j in labels_o])/len(labels_o)
                    if b_o < best_distance:
                        best = o
                        best_distance = b_o

        return best_distance
    
    
    def constrained_silhouette(self, Distances, cluster):
        # for an input distance matrix Distances and an input cluster (list of items)
        # compute a silhouette score constrained by the requirement that each item  
        # can only join alternative clusters if doing so satisfies clique constraints
        
        silhouette_score = 0
        for i in self.clustering[cluster]:
            labels_i = np.where(self.labels == self.labels[i])[0]
            if len(labels_i) != 1:
                a_i = np.sum([Distances[i,j] if i < j else Distances[j,i] for j in labels_i if j != i])/(len(labels_i) - 1)
                b_i = self.find_closest_other(Distances, i)

                if b_i != np.inf:
                    silhouette_score += (b_i - a_i)/max(a_i, b_i)
                else:
                    silhouette_score += 0
            
        return silhouette_score/len(self.clustering[cluster])
    
    
    def silhouette(self, cluster):
        # this is just a wrapper for constrained silhouette that allows me to 
        # compute using the initialized distances self.D
        return self.constrained_silhouette(self.D, cluster)
    
        
        
    def total_points(self, cluster):
        # Compute the total number (location, time) points 
        # covered by items of a cluster
        total_points = 0
        for k in cluster:
            times = self.segment_pool.times[self.segment_pool.key_list[k]]
            total_points += times[1] - times[0]
            
        return total_points
    
    def explained_variance(self, cluster):
        # after finding the subset of data covered by items in an input cluster,
        # compute the amount of variance seen within that subset of data 
        X = self.segment_pool.data.to_numpy()
        X = X - np.mean(X, axis = 0)
        X = X**2
            
        explained = 0
        for k in cluster:
            loc = self.segment_pool.key_list[k]
            iloc = self.segment_pool.data.columns.get_loc(loc[:loc.rfind('_')])
            times = self.segment_pool.times[loc]
            explained += np.sum(X[times[0]:times[1], iloc])
                
        return explained
    
    

    def avg_values_time(self, cluster, time_series_data):
        # For an input set of time-series data with shape identical to the time-series data 
        # associated with the segment_pool object, compute the average 
        # value seen within the subset of data covered by cluster
        clust_vals = []
        for k in range(len(cluster)):
            k_name = self.segment_pool.key_list[cluster[k]]
            k_times = self.segment_pool.times[k_name]
            k_vec = time_series_data.loc[:,k_name[:k_name.rfind('_')]][k_times[0]:k_times[1]]
            k_val = k_vec.mean()
            clust_vals.append(k_val)
            
        return np.mean(clust_vals)
    
    
    
######################################################################################################################
## Tools for analyzing a clustering with some auxiliary information! 
# This is specifically catered to the experiments I was performing
######################################################################################################################

def compute_auxiliary_info(analysis_object, dist_matrices, health_index, infection_data, extra_info):
    cluster_data = []
    for clust in range(len(analysis_object.clustering)):
        cluster = analysis_object.clustering[clust]
        size = len(cluster)
        cost = analysis_object.complete_linkage_cost(cluster)
        points = analysis_object.total_points(cluster)
        explained_variance = analysis_object.explained_variance(cluster) 

        silhouettes = [analysis_object.silhouette(clust)]
        for Dm in dist_matrices:
            sil = analysis_object.constrained_silhouette(Dm, clust)
            silhouettes.append(sil)
            
        pairwise = []
        for Dm in dist_matrices:
            score = analysis_object.avg_pairwise_distances(Dm, clust)
            pairwise.append(score)
            
        health_info = None
        if not health_index is None:
            health_info = analysis_object.avg_values_time(cluster, health_index)
        
        infection_info = None
        if not infection_data is None:
            infection_info = analysis_object.avg_values_time(cluster, infection_data)
            
        cluster_data.append([i for i in extra_info] + [clust, size, cost, points, explained_variance] + silhouettes + pairwise + [health_info, infection_info])
        
    return cluster_data

def auxiliary_info(cluster_object, wpool, dist_matrices, health_index = None, infection_data = None, extra_info = []):
    c_analyze = cluster_analyze(cluster_object, segment_pool = wpool)
    cluster_data = compute_auxiliary_info(c_analyze, dist_matrices, health_index, infection_data, extra_info = extra_info)
        
    return cluster_data