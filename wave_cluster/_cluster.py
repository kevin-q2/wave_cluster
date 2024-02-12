import numpy as np
import scipy.sparse
import networkx as nx
import itertools 
import time
import math
import random
import copy

##############################################################################
#   Clique clustering implementation
#   Kevin Quinn 11/23
#
#
##############################################################################

### TO DO: add node list into this (affects number of clusters!)

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
                
            if boo == False:
                #print("Can't find less than " + str(len(self.C)) + " clusters!")
                #print('clique time: ' + str(clique_time))
                #print('dist time: ' + str(dist_time))
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
            
            

##################################################################################################
#
###################################################################################################

class cluster_mcmc:
    def __init__(self, clustering, node_list, edge_list):
        self.k = len(clustering)
        self.n = len(node_list)
        self.current_clustering = clustering
        self.current_labels = self.cluster_to_label(self.current_clustering)
        self.G = nx.Graph()
        self.G.add_nodes_from(node_list)
        self.G.add_edges_from(edge_list)
        #self.random_clusterings = [self.cl]uster_to_label(clustering)
        
        
    def cluster_to_label(self, clustering):
        lens = [len(i) for i in clustering]
        labels = np.zeros(np.sum(lens), dtype = np.int8)
        for c in range(len(clustering)):
            for k in clustering[c]:
                labels[k] = c
                
        return labels

    def check_connect(self, element, c_i):
        for i in self.current_clustering[c_i]:
            if not self.G.has_edge(i,element):
                return False
        return True
    
    def expand(self):
        rand = np.random.choice(range(self.k))
        for c in range(self.k):
            if c != rand and len(self.current_clustering[c]) > 1:
                for n in self.current_clustering[c]:
                    if self.check_connect(n, rand):
                        if np.random.uniform(0,1) > 0.5:
                            #print('expansion!')
                            self.current_clustering[c].remove(n)
                            self.current_clustering[rand].append(n)
    
    def split(self):
        rand = np.random.choice(range(self.k))
        old_cluster = []
        new_cluster = []
        for n in self.current_clustering[rand]:
            if np.random.uniform(0,1) > 0.5:
                new_cluster.append(n)
            else:
                old_cluster.append(n)
                
        self.current_clustering[rand] = old_cluster
        self.current_clustering.append(new_cluster)
        
    
    def permutations(self, clustering):
        cluster_moves = 0
        clustering_moves = 1
        spaces = 0
        for ci in range(len(clustering)):
            
            # I think don't add anything for the case of singleton cliques???
            if len(clustering[ci]) > 1:
                prod = 1
                for ell in range(2, len(clustering[ci]) + 1):
                    prod *= math.comb(ell, 2)
                cluster_moves += prod
            
                #print(spaces)
                spaces += len(clustering[ci]) - 1
                clustering_moves *= math.comb(spaces, len(clustering[ci]) - 1)
                #spaces = spaces - (len(self.current_clustering) - 1)

        #print('clustering: ' + str(clustering_moves))
        #print('cluster: ' + str(cluster_moves))
        return clustering_moves * cluster_moves
        
    def single_move(self):
        v = np.random.choice(range(len(self.G.nodes)))
        current_cluster = self.current_labels[v]
        # Is there a more efficient way to do this?
        Q = [i for i in range(len(self.current_clustering)) if self.check_connect(v, i)]
        
        if len(Q) == 0 or len(self.current_clustering[current_cluster]) == 1:
            return
        
        move_to = np.random.choice(Q)
        attempt_clustering  = copy.deepcopy(self.current_clustering)
        attempt_clustering[current_cluster].remove(v)
        attempt_clustering[move_to].append(v)
        
        u = np.random.uniform(0,1)
        #print(self.permutations(attempt_clustering))
        #print('total: ' +  str(self.permutations(attempt_clustering)))
        #print()
        #print('total: ' + str(self.permutations(self.current_clustering)))
        #print()
        f_ratio = self.permutations(attempt_clustering)/self.permutations(self.current_clustering)
        #print(f_ratio)
        #print(u)
        #print()
        if f_ratio >= u:
            self.current_clustering = attempt_clustering
        
    
    def sample(self, num_samples):
        self.random_clusterings = np.zeros((num_samples, len(self.G.nodes)))
        for s in range(num_samples):
            #self.expand()
            #if len(self.current_clustering) < self.k:
            #    print('split')
            #    self.split()
            self.single_move()
            self.current_labels = self.cluster_to_label(self.current_clustering)
            self.random_clusterings[s,:] = self.current_labels
            
        self.random_clusterings = np.array(self.random_clusterings)
            
        
    
    
    
            
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
        
    
    def cluster_avg_diameter_adj(self, cluster, Graph):
        if len(cluster) != 1:
            clust_dists = []
            for k in cluster:
                k_name = self.segment_pool.key_list[k][:5]
                for k2 in cluster:
                    k2_name = self.segment_pool.key_list[k2][:5]
                    if k != k2:
                        dist = nx.shortest_path_length(Graph, k_name, k2_name)
                        clust_dists.append(dist)
                        
            return np.sum(clust_dists)/(len(cluster) * (len(cluster) - 1))
        else:
            return 0
        
    
    def clustering_avg_diameter(self, Graph):
        diameters = np.zeros(len(self.clustering))
        
        for ci in range(len(self.clustering)):
            avg_c = self.cluster_avg_diameter(self.clustering[ci], Graph)
            diameters[ci] = avg_c
            
        return diameters
        
        
    
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
        
    