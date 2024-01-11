import numpy as np
import scipy.sparse
import networkx as nx
import itertools 
import time
import math

##############################################################################
#   Clique clustering implementation
#   Kevin Quinn 11/23
#
#
##############################################################################



class clique_cluster:
    def __init__(self, D, edge_list, linkage = 'single'):
        self.D = D
        self.G = nx.from_edgelist(edge_list)
            
        self.C = [[i] for i in range(D.shape[1])]
        self.D_cluster = D.copy()
        
        if linkage == 'single':
            self.linkage = self.single_linkage
        elif linkage == 'complete':
            self.linkage = self.complete_linkage
            
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
                linkage_val = self.linkage(self.C[i], self.C[j])
                
                '''
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
                
                    '''        
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
                
            if boo == False:
                print("Can't find less than " + str(len(self.C)) + " clusters!")
                print('clique time: ' + str(clique_time))
                print('dist time: ' + str(dist_time))
                return
            
            #print(min_r)
            #print(min_c)
            #print()
            self.cluster_update(min_r, min_c, v[min_v])
            start_time = time.time()
            self.distance_update()
            end_time = time.time()
            dist_time += end_time - start_time
    
           
    def random_cluster(self,k):
        while len(self.C) > k:
            randlist = list(itertools.combinations(range(len(self.C)), 2))
            order = np.random.choice(range(len(randlist)), len(randlist))
            boo = False
            start = 0
            while not boo and start < len(order):
                rand_r = randlist[order[start]][0]
                rand_c = randlist[order[start]][1]
                boo = self.check_clique(rand_r, rand_c)
                start += 1
                
            if boo == False:
                print("Can't find less than " + str(len(self.C)) + " clusters!")
                return
            
            self.cluster_update(rand_r, rand_c, 0)
            
            
            
            
#################################################################################################
#   
## using a vector of cluster labels (n dimensional vector where each entry is 
    # a label from [1,..,k]) compute the clustering partition (k dimensional list of lists 
    # where each element is assigned to its cluster)
################################################################################################
            
            
class cluster_analyze:
    def __init__(self, clustering = None, labels = None, segment_pool = None):
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
            
        self.segment_pool = segment_pool

            
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
    
    
    def clust_nearest_neighbor(self, c, cluster, Graph):
        complete = False
        if math.comb(len(Graph.nodes),2) == len(Graph.edges):
            complete = True
        
        c_name = self.segment_pool.key_list[c][:5]
        min_dist = np.inf
        neighbor = None
        for i in cluster:
            if i != c:
                i_name = self.segment_pool.key_list[i][:5]
                if complete:
                    dist = Graph.get_edge_data(c_name, i_name)['weight']
                else:
                    try:
                        dist = nx.shortest_path_length(Graph, c_name, i_name) #, weight = 'weight')
                        #dist = nx.path_weight(Graph, path, weight = 'weight')
                    except:
                        dist = np.inf

                if dist < min_dist:
                    min_dist = dist
                    neighbor = i
                    
        return neighbor, min_dist
    
    
    
    def cluster_cohesive(self, cluster, Graph):
        if len(cluster) != 1:
            clust_dists = []
            for k in cluster:
                nearest_dist = self.clust_nearest_neighbor(k, cluster, Graph)[1]
                if nearest_dist != np.inf:
                    clust_dists.append(nearest_dist)

            return np.mean(clust_dists)
        else:
            return 0
        
        
    def cluster_avg_diameter(self, cluster, Graph):
        if len(cluster) != 1:
            clust_dists = []
            for k in cluster:
                k_name = self.segment_pool.key_list[k][:5]
                for k2 in cluster:
                    k2_name = self.segment_pool.key_list[k2][:5]
                    if k != k2:
                        dist = Graph.get_edge_data(k_name, k2_name)['weight']
                        clust_dists.append(dist)

            return np.sum(clust_dists)/(len(cluster) * (len(cluster) - 1))
        else:
            return 0
        
    
    def clustering_cohesive(self, Graph):
        clustering_dists = []
        for clust in self.clustering:
            clust_dists = []
            if len(clust) != 1:
                for k in clust:
                    nearest_dist = self.clust_nearest_neighbor(k, clust, Graph)[1]
                    if nearest_dist != np.inf:
                        clust_dists.append(nearest_dist)
                clustering_dists.append(np.mean(clust_dists))
    
        return np.mean(clustering_dists)
        #return np.mean(np.sort(clustering_dists)[:10])
        
    