# -*- coding: utf-8 -*-
"""
Package for implmenting Gower's method for finding latent networks in multi-modal data

author: Iain Cruickshank
license: MIT version 3
acknowledgements: Work supported by National Science Foundation Graduate Research Fellowship (DGE 1745016).   
"""

import numpy as np, networkx as nx
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sknetwork.clustering import Louvain, modularity

class graph_computation:
    """Core class that handles simialrity calculations and graph learning
    
    Attributes
    ----------
    gowers_scheme : str or list, optional
        The type of weighting scheme to use. Default is entropy. Use 'unweighted'
        for no weights. Pass a list of custom weights for using custom weights;
        note the length of the list must be the same as the number of columns across
        all modes of the data.
    construction_method : str, optional
        the type of graph construction method to use. Current implemented options
        are the modularity k-NN, denoted as 'modularity' which is the default and the
        weighted consensus graph which is denoted as 'WCG'.
    WCG_epsilon : float, optional
        specify the threshold for not including edges in the weighted_consensus_graph.
        default is 0.01
    knn_symmetrize : boolean, optional
        determine wether you want to use symmetric (link exists only if it is mutual)
        or assymetric kNNs (link exists if it is present in any direction) for the kNN modualrity
        method. Default is assymetric.
    network_enhancement: boolean, optional
        wether to perform the diffusion process known as Network Enhancement 
        following fitting an approxiamate graph to the data.
        
    Methods
    -------
    compute_similarity(m)
        compute the pairwise simialrity between all entities using Gower's Coefficient
        of Similarity (weighted or unweighted)
    compute_graph(A)
        Learn the graph from the defined construction method on a pairwise similarity
        matrix, A.
    """
    
    def __init__(self, gowers_scheme, construction_method, 
                 WCG_epsilon, knn_graph_type, network_enhancement, enhancement_iterations, 
                 enhancement_alpha, enhancement_nearest_neighbors):
        self.gowers_scheme = gowers_scheme
        self.epsilon = WCG_epsilon
        self.graph_type = knn_graph_type
        self.construction_method = construction_method
        self.network_enhancement = network_enhancement
        self.T = enhancement_iterations
        self.alpha = enhancement_alpha
        self.nearest_neighbors = enhancement_nearest_neighbors
    
    
    def compute_similarity(self, m):
        
        if len(m) > 1:
            X =np.zeros((m[0].shape[0], m[0].shape[0]))
            weights =[]

            for mode in range(len(m)):
                mode_weights = self._construct_mode_weights(m[mode], 
                                                                 weight_scheme=self.gowers_scheme)
                weights.append(mode_weights)
        else:
            X =np.zeros((m.shape[0], m.shape[0]))
            weights = self._construct_mode_weights(m, weight_scheme=self.gowers_scheme)
            
        N = X.shape[0]
        X = np.array([self._similarity_computations(i,m,N,weights) for i in range(N)])
                
        self.X = X + np.transpose(X)
        return self.X
 
    
    def _similarity_computations(self, i, m, N, weights):
        X=np.zeros(N)
        for j in range(i+1,N):
                if len(m) > 1:
                    unweighted_values = []
                    append = unweighted_values.append
                    for mode in range(len(m)):
                        if np.issubdtype(m[mode].dtype, np.number):
                            r = np.ptp(m[mode], axis=0)
                            numerator = np.abs(m[mode][i,:] - m[mode][j,:], dtype=float)
                            append(1-np.divide(numerator, r, out=np.zeros_like(numerator), where=r!=0))
                        else:
                            append(np.multiply((m[mode][i,:] !='0') == (m[mode][j,:]!='0'), np.ones(len(m[mode][i,:]))))
                    X[j] = np.inner(np.concatenate(unweighted_values).ravel(), np.concatenate(weights).ravel())/np.sum(np.concatenate(weights).ravel())
                else:
                    if np.issubdtype(m.dtype, np.number):
                        r = np.ptp(m, axis=0)
                        numerator = np.abs(m[i,:] - m[j,:], dtype=float)
                        unweighted_values = 1-np.divide(numerator, r, out=np.zeros_like(numerator), where=r!=0)
                    else:
                        unweighted_values = np.multiply((m[i,:] !='0') == (m[j,:] !='0'), np.ones(len(m[i,:])))
                    X[j] = np.inner(np.array(unweighted_values), np.array(weights))/np.sum(np.array(weights))

        return X        
        
    
    def _construct_mode_weights(self, m, weight_scheme='entropy'):
        h,w = m.shape
        A = np.zeros((h+w, h+w))
        A[:h, :w] = m
        A[h:, w:] = np.transpose(m)
        if type(weight_scheme) is not str:
            return weight_scheme
        elif weight_scheme.lower() == 'entropy':
            N= A.shape[0]
            weights=np.ones(m.shape[1])
            D= np.sum(A,1)*np.eye(N)
            with np.errstate(all='ignore'):
                D_inv = np.nan_to_num(np.power(np.sum(A,1),-0.5)*np.eye(N))
                L = np.nan_to_num(np.matmul(D_inv,np.matmul(D-A, D_inv)))
                eigenvalues = np.real(np.linalg.eig(L)[0])
                eigenvalues[eigenvalues<1e-9] =0 
                eigenvalues = eigenvalues[eigenvalues.nonzero()]
                entropy = -1* np.sum(np.nan_to_num(np.multiply(eigenvalues/N, np.log(eigenvalues/N))))
            return weights * entropy
        else:
            '''
            Use for no weights
            '''
            return np.ones(m.shape[1])
        
    
    def compute_graph(self, A):
        if self.construction_method == 'WCG':
            final_graph = self._weighted_consensus_graph(A)
        else:
            final_graph= self._kNN_modularity(A)
            
        if self.network_enhancement:
            return self._network_enhancement(final_graph)
        else:
            return final_graph
        
    
    def _weighted_consensus_graph(self, A):
        N = A.shape[0]
        C = np.zeros((N,N))
        k = np.floor(0.4*N)
        
        idxs = []
        for i in range(N):
            nonzero_entries = np.nonzero(A[i,:])[0]
            sorted_nonzero_idxs = np.argsort(A[i,nonzero_entries])
            idxs.append(nonzero_entries[sorted_nonzero_idxs])
        
        for u in range(N):
            neighbors = idxs[u][0:k]
            for v in range(N):
                for w in range(v+1,N):
                    if (v in neighbors) and (w in neighbors):
                        N_v = np.zeros(N)
                        N_v[idxs[v][0:k]] = A[v,idxs[v][0:k]]
                        N_w = np.zeros(N)
                        N_w[idxs[w][0:k]] = A[v,idxs[w][0:k]]
                        edge_prob = np.sum(np.minimum(N_w,N_v))/np.sum(np.maximum(N_w,N_v))
                        if edge_prob >= self.epsilon:
                            C[w,v] += edge_prob
                            C[v,w] += edge_prob
        
        return normalize(C, norm='l1', axis=1, copy=False)
    
    
    def _kNN_modularity(self, A):
        best_modularity =-1
        best_network = nx.Graph()
        best_k =2
        distances = 1-A
        distances[distances < 1e-10] =0
    
        for n in range(1, np.int(np.floor(np.log2(distances.shape[0])))):
            k = 2**n
            nbrs = NearestNeighbors(n_neighbors=k, metric='precomputed').fit(distances)
            kNN = nbrs.kneighbors_graph(distances)
            if self.graph_type == 'symmetric':
                kNN = kNN.minimum(kNN.T)
            elif self.graph_type=='assymetric':
                kNN = kNN.maximum(kNN.T)
            network = nx.from_scipy_sparse_matrix(kNN)
            current_clusters, avg_modularity = self._get_partition_and_modularity(kNN)
            S = network.number_of_nodes()
            p = nx.density(network)
            random_modularity = (1-2/np.sqrt(S))*(2/(p*S))**(2/3)
            current_modularity = avg_modularity - random_modularity

            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_network = network
                best_k = k

        print("The modularity of the best learned graph was: {}, at a k-value of: {}".format(best_modularity, best_k))
        return nx.to_numpy_array(best_network)
    
    def _get_partition_and_modularity(self, kNN):
        cluster_iterations = [Louvain(random_state=int(rand_state), shuffle_nodes=True) for rand_state in np.random.randint(1,high=100000,size=10)]
        parts = [clstr.fit(kNN).labels_ for clstr in cluster_iterations]
        modularities = [modularity(kNN, part) for part in parts]
        best_part = parts[np.argmax(modularities)]
        avg_modularity = np.mean(modularities)
        return best_part, avg_modularity
    
    
    def _network_enhancement(self, full_graph, alpha=0.9):
        
        if self.nearest_neighbors == 'sqrt':
            nearest_neighbors = int(np.ceil(np.sqrt(full_graph.shape[0])))
        else:
            nearest_neighbors = self.nearest_neighbors

        D = np.diag(np.array(np.sqrt(np.sum(full_graph, axis=1))).ravel())
        N = full_graph.shape[0]
        Q = self._DSM(full_graph)
        idxs = np.flip(np.argsort(Q, axis=1), axis=1)[:,1:nearest_neighbors+1]
        graph = np.zeros((N,N))
        graph[np.arange(N)[:,None], idxs] = Q[np.arange(N)[:,None], idxs]
        self.kernel_graph = self._DSM(graph)
        
                
        for t in range(self.T+1):
            Q = alpha*(self.kernel_graph @ Q @ self.kernel_graph) + (1-alpha)*self.kernel_graph
            
        np.fill_diagonal(Q,0)
        Q = D @ Q @ D
        return Q  
    
    
    def _DSM(self, graph):
        graph = normalize(graph, norm='l1', axis=1, copy=False)
        col_sums= np.sum(graph, axis=0)
        temp_graph = np.divide(graph, col_sums**0.5, out=np.zeros_like(graph), where=col_sums!=0)
        return temp_graph @ temp_graph.T
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        