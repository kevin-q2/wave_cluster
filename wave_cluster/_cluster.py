import numpy as np
import scipy.sparse
import networkx as nx
import itertools 
import time
import math
import random
import copy
#from sklearn.metrics import silhouette_score
#from ._tools import constrained_silhouette 

##############################################################################
#   Clique clustering implementation
#   Kevin Quinn 11/23
#
#
##############################################################################

### TO DO: add node list into this (affects number of clusters!)

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
        
        self.linkage_matrix = []
        self.linkage_index = len(self.C) - 1
        self.linkage_names = [i for i in range(D.shape[1])]
        
        
            
    # compute the single linkage distance between clusters A and B
    def single_linkage(self, A,B):
        min_dist = np.inf
        for a in A:
            for b in B:
                r = min(a,b)
                c = max(a,b)
                Dij = self.D[r, c]
                if Dij < min_dist:
                    min_dist = Dij
        return min_dist
    
    # compute the complete linkage distance between clusters A and B
    def complete_linkage(self, A,B):
        max_dist = 0
        for a in A:
            for b in B:
                r = min(a,b)
                c = max(a,b)
                Dij = self.D[r, c]
                if Dij > max_dist:
                    max_dist = Dij
        return max_dist
    
    def distance_update(self):
        ro,co,vo = scipy.sparse.find(self.D_cluster)
        rdx = []
        cdx = []
        vals = []
        for i in range(len(self.C) - 1):
            for j in range(i + 1, len(self.C)):
                # newly formed cluster appears at the end of cluster list
                #linkage_val = self.linkage(self.C[i], self.C[j])
                
                
                if j == len(self.C) - 1:
                    linkage_val = self.linkage(self.C[i], self.C[j])
                
                
                # for non-newly formed clusters, we already have the distance between them (no need to recompute)
                else:
                    if i >= self.last_links[1] - 1:
                        diff_i = i + 2
                    elif i >= self.last_links[0]:
                        diff_i = i + 1
                    else:
                        diff_i = i
                        
                    if j >= self.last_links[1] - 1:
                        diff_j = j + 2
                    elif j >= self.last_links[0]:
                        diff_j = j + 1
                    else:
                        diff_j = j
                        
                        
                    linkage_val = self.D_cluster[diff_i, diff_j]
                
                
                            
                rdx.append(i)
                cdx.append(j)
                vals.append(linkage_val)
        
        self.D_cluster = scipy.sparse.csr_matrix((vals, (rdx, cdx)), shape = (len(self.C),len(self.C)))
            
    
    def cluster_update(self, c_i, c_j, v_ij):
        new_cluster = [self.C[c_i] + self.C[c_j]]
        self.C = [self.C[i] for i in range(len(self.C)) if i != c_i and i != c_j] + new_cluster
        self.last_links = [c_i, c_j]
        
        #rd, cd, vd = scipy.sparse.find(self.D_cluster)
        #rdx = [self.C[i] for i in range(len(self.C)) if i != c_i and i != c_j]
        
        self.linkage_index += 1
        self.linkage_matrix.append([self.linkage_names[c_i], self.linkage_names[c_j], v_ij])
        self.linkage_names = [self.linkage_names[i] for i in range(len(self.linkage_names)) if i != c_i and i != c_j] + [self.linkage_index]
    
    
    def check_clique(self, c_i, c_j):
        for i in self.C[c_i]:
            for j in self.C[c_j]:
                if not self.G.has_edge(i,j):
                    return False
        return True 
    
    
    
    def cluster(self, k):
        clique_time = 0
        dist_time = 0
        while len(self.C) > k:
            r,c,v = scipy.sparse.find(self.D_cluster)
            v_sort = np.argsort(v)
            start = 0
            boo = False
            while not boo and start < len(v):
                min_v = v_sort[start]
                min_r = r[min_v]
                min_c = c[min_v]
                start_time = time.time()
                boo = self.check_clique(min_r, min_c)
                end_time = time.time()
                clique_time += end_time - start_time
                
                start += 1
            
            # No more cliques or distance threshold reached!
            if boo == False:
                #print("Can't find less than " + str(len(self.C)) + " clusters!")
                #print('clique time: ' + str(clique_time))
                #print('dist time: ' + str(dist_time))
                return
            
            elif start < len(v):
                if v[v_sort[start]] > self.threshold:
                    return
            
            #print(min_r)
            #print(min_c)
            #print()
            self.cluster_update(min_r, min_c, v[min_v])
            start_time = time.time()
            self.distance_update()
            end_time = time.time()
            dist_time += end_time - start_time
    

###############################################################################################################################

##############################################################################################################################
            
            
            
class random_clique_cluster(clique_cluster):
    def __init__(self, node_list, edge_list):
        self.edge_list = edge_list
        self.node_list = node_list
        self.G = nx.from_edgelist(edge_list)
        self.C = [[i] for i in range(len(node_list))]
        self.D = np.zeros((len(node_list), len(node_list)))
               
    def cluster_update(self, c_i, c_j, v_ij):
        new_cluster = [self.C[c_i] + self.C[c_j]]
        self.C = [self.C[i] for i in range(len(self.C)) if i != c_i and i != c_j] + new_cluster

    '''
    def random_combination(self, iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)
    '''
           
    def cluster(self,k):
        while len(self.C) > k:
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
                randlist = list(itertools.combinations(range(len(self.C)), 2))
                order = np.random.choice(range(len(randlist)), len(randlist))
                start2 = 0
                while not boo and start2 < len(order):
                    rand_r = randlist[order[start2]][0]
                    rand_c = randlist[order[start2]][1]
                    boo = self.check_clique(rand_r, rand_c)
                    start2 += 1
                    
                if boo == False:
                    #print("Can't find less than " + str(len(self.C)) + " clusters!")
                    return
            
            self.cluster_update(rand_r, rand_c, 0)
            
    
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
            
        #random_clusterings = random_clusterings[np.max(random_clusterings, axis = 1) == k - 1]
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
        
    
    
    
            
#################################################################################################
#   
## using a vector of cluster labels (n dimensional vector where each entry is 
    # a label from [1,..,k]) compute the clustering partition (k dimensional list of lists 
    # where each element is assigned to its cluster)
################################################################################################
            
            
class cluster_analyze:
    #def __init__(self, D, segment_pool, clustering = None, labels = None, edgelist = None):
    def __init__(self, cluster_object, segment_pool):
        self.cluster_object = cluster_object
        self.segment_pool = segment_pool
        self.D = self.cluster_object.D
        self.G = self.cluster_object.G
        self.clustering = self.cluster_object.C
        self.labels = self.cluster_to_label(self.clustering)
        
        '''
        if clustering is None and labels is None:
            pass
        elif labels is None:
            self.clustering = clustering 
            self.labels = self.cluster_to_label(clustering)
            
        elif clustering is None:
            self.labels = labels 
            self.clustering = self.label_to_cluster(labels)
            
        else:
            self.clustering = clustering
            self.labels = labels
            
        if not edgelist is None:
            self.G = nx.from_edgelist(edgelist)
        else:
            self.G = None
        '''

            
    def cluster_to_label(self, clustering):
        lens = [len(i) for i in clustering]
        labels = np.zeros(np.sum(lens))
        
        for c in range(len(clustering)):
            for k in clustering[c]:
                labels[k] = c
                
        return labels
        

    def label_to_cluster(self, labels):
        clustering = [[] for i in range(int(np.max(labels)) + 1)]
        for l in range(len(labels)):
            clustering[int(labels[l])].append(l)
            
        return clustering
    
    
    def avg_cost(self, cluster):
        if len(cluster) != 1:
            clust_costs = []
            for k in range(len(cluster)):
                for k2 in range(k + 1, len(cluster)):
                    if cluster[k] < cluster[k2]:
                        cost = self.D[cluster[k],cluster[k2]]
                    else:
                        cost = self.D[cluster[k2],cluster[k]]
                        
                    clust_costs.append(cost)

            return np.mean(clust_costs)
        else:
            return 0
        
    
    def complete_linkage_cost(self, cluster):
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
        others = np.unique(self.labels)
        best = None
        best_distance = np.inf
        for o in others:
            if o != self.labels[point]:
                labels_o = np.where(self.labels == o)[0]
                
                overlaps_with = [k for k in labels_o if self.G.has_edge(point, k)]
                if len(overlaps_with) >= 1*len(labels_o):
                    b_o = np.sum([Distances[point,j] if point < j else Distances[j,point] for j in labels_o])/len(labels_o)
                    #print(b_o)
                    if b_o < best_distance:
                        best = o
                        best_distance = b_o

        return best_distance


    '''
    def constrained_silhouette(self, Distances):
        silhouette_score = 0
        for i in range(len(self.labels)):
            labels_i = np.where(self.labels == self.labels[i])[0]
            if len(labels_i) != 1:
                a_i = np.sum([Distances[i,j] if i < j else Distances[j,i] for j in labels_i if j != i])/(len(labels_i) - 1)
                b_i = self.find_closest_other(Distances, i)

                if b_i != np.inf:
                    silhouette_score += (b_i - a_i)/max(a_i, b_i)
                else:
                    silhouette_score += 1
            
        return silhouette_score/len(self.labels)
    '''
    
    def constrained_silhouette(self, Distances, cluster):
        silhouette_score = 0
        for i in self.clustering[cluster]:
            labels_i = np.where(self.labels == self.labels[i])[0]
            if len(labels_i) != 1:
                a_i = np.sum([Distances[i,j] if i < j else Distances[j,i] for j in labels_i if j != i])/(len(labels_i) - 1)
                b_i = self.find_closest_other(Distances, i)

                if b_i != np.inf:
                    silhouette_score += (b_i - a_i)/max(a_i, b_i)
                else:
                    silhouette_score += 1
            
        return silhouette_score/len(self.clustering[cluster])
    
    
    def silhouette(self, cluster):
        return self.constrained_silhouette(self.D, cluster)
        
        
    def total_points(self, cluster):
        total_points = 0
        for k in cluster:
            times = self.segment_pool.times[self.segment_pool.key_list[k]]
            total_points += times[1] - times[0]
            
        return total_points
    
    def explained_variance(self, cluster):
        X = self.segment_pool.data.to_numpy()
        X = X - np.mean(X, axis = 0)
        #X = X - np.mean(X)
        X = X**2
            
        explained = 0
        for k in cluster:
            loc = self.segment_pool.key_list[k]
            iloc = self.segment_pool.data.columns.get_loc(loc[:loc.rfind('_')])
            times = self.segment_pool.times[loc]
            explained += np.sum(X[times[0]:times[1], iloc])
                
        return explained

        
    def avg_pairwise_distance(self, cluster, Graph):
        if len(cluster) != 1:
            clust_dists = []
            for k in range(len(cluster)):
                k_name = self.segment_pool.key_list[cluster[k]]
                k_name = k_name[:k_name.rfind('_')]
                for k2 in range(k + 1, len(cluster)):
                    k2_name = self.segment_pool.key_list[cluster[k2]]
                    k2_name = k2_name[:k2_name.rfind('_')]
                    dist = Graph.get_edge_data(k_name, k2_name)['weight']
                    clust_dists.append(dist)

            return np.mean(clust_dists)
        else:
            return 0
        
    
    def avg_pairwise_distance_adj(self, cluster, Graph):
        if len(cluster) != 1:
            clust_dists = []
            for k in range(len(cluster)):
                k_name = self.segment_pool.key_list[cluster[k]]
                k_name = k_name[:k_name.rfind('_')]
                for k2 in range(k+1, len(cluster)):
                    k2_name = self.segment_pool.key_list[cluster[k2]]
                    k2_name = k2_name[:k2_name.rfind('_')]
                    dist = nx.shortest_path_length(Graph, k_name, k2_name)
                    clust_dists.append(dist)
                        
            return np.mean(clust_dists)
        else:
            return 0
        
    
    def avg_pairwise_distance_time(self, cluster, time_series_data):
        if len(cluster) != 1:
            clust_dists = []
            for k in range(len(cluster)):
                k_name = self.segment_pool.key_list[cluster[k]]
                k_times = self.segment_pool.times[k_name]
                k_vec = time_series_data.loc[:,k_name[:k_name.rfind('_')]][k_times[0]:k_times[1]]
                k_val = k_vec.mean()
                for k2 in range(k+1, len(cluster)):
                    k2_name = self.segment_pool.key_list[cluster[k2]]
                    k2_times = self.segment_pool.times[k2_name]
                    k2_vec = time_series_data.loc[:,k2_name[:k2_name.rfind('_')]][k2_times[0]:k2_times[1]]
                    k2_val = k2_vec.mean()
                    
                    dist = np.abs(k_val - k2_val)
                    clust_dists.append(dist)
                        
            return np.mean(clust_dists)
        else:
            return 0
        

    def avg_values_time(self, cluster, time_series_data):
        clust_vals = []
        for k in range(len(cluster)):
            k_name = self.segment_pool.key_list[cluster[k]]
            k_times = self.segment_pool.times[k_name]
            k_vec = time_series_data.loc[:,k_name[:k_name.rfind('_')]][k_times[0]:k_times[1]]
            k_val = k_vec.mean()
            clust_vals.append(k_val)
            
        return np.mean(clust_vals)
    
    
## Tools for analyzing a clustering with some auxiliary information! 
# This is specifically catered to the experiments I was performing
def compute_auxiliary_info(analysis_object, dist_matrices, health_index, infection_data, extra_info):
    cluster_data = []
    for clust in range(len(analysis_object.clustering)):
        cluster = analysis_object.clustering[clust]
        size = len(cluster)
        cost = analysis_object.complete_linkage_cost(cluster)
        points = analysis_object.total_points(cluster)
        explained_variance = analysis_object.explained_variance(cluster) 
        
        #diams = []
        silhouettes = [analysis_object.silhouette(clust)]
        for Dm in dist_matrices:
            #diam = analysis_object.avg_pairwise_distance(cluster, dG)
            #diams.append(diam)
            sil = analysis_object.constrained_silhouette(Dm, clust)
            silhouettes.append(sil)
            
        health_info = None
        if not health_index is None:
            health_info = analysis_object.avg_values_time(cluster, health_index)
            #health_dists = analysis_object.avg_pairwise_distance_time(cluster, health_index)
        
        infection_info = None
        if not infection_data is None:
            infection_info = analysis_object.avg_values_time(cluster, infection_data)
            
        #if compute_silhouette:
        #    silhouette_score = analysis_object.silhouette()
        #else:
        #    silhouette_score = None
        #cluster_data.append([i for i in extra_info] + [clust, size, cost, silhouette_score, points, explained_variance] + diams + [health_info, health_dists, infection_info])
        cluster_data.append([i for i in extra_info] + [clust, size, cost, points, explained_variance] + silhouettes + [health_info, infection_info])
        
    return cluster_data

def auxiliary_info(cluster_object, wpool, dist_matrices, health_index = None, infection_data = None, extra_info = []):
    # analysis object
    c_analyze = cluster_analyze(cluster_object, segment_pool = wpool)
    cluster_data = compute_auxiliary_info(c_analyze, dist_matrices, health_index, infection_data, extra_info = extra_info)
        
    return cluster_data