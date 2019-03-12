# -*- coding: utf-8 -*-
"""
Package for implmenting Gower's method for finding latent networks in multi-modal data

author: Iain Cruickshank
license: MIT version 3
acknowledgements: Work supported by National Science Foundation Graduate Research Fellowship (DGE 1745016).   
"""

import numpy as np, pandas as pd, networkx as nx, community, multiprocessing as mp

class latent_graph:
    """Finds a latent graph of data using Gower's method and specified graph constructor
    
    Attributes
    ----------
    gowers_scheme : str or list, optional
        The type of weighting scheme to use. Default is entropy. Use 'unweighted'
        for no weights. Pass a list of custom weights for using custom weights;
        note the length of the list must be the same as the number of columns across
        all modes of the data.
    epsilon : float, optional
        specify the threshold for not including edges in the weighted_consensus_graph.
        default is 0.01
    construction_method : str, optional
        the type of graph construction method to use. Current implemented options
        are the modularity k-NN, denoted as 'modularity' which is the default and the
        weighted consensus graph which is denoted as 'WCG'.
        
    Methods
    -------
    load_data_from_file(data, remove_non_distinguishing=False)
        Read in raw data from a filename or list of file names
    load_data_from_memory(data, remove_non_distinguishing=False)
        Read in raw data already in memory and convert to the format expected by the methods
    learn_graph()
        Function call to do Gower's method on the read-in raw data and return the latent network
    return_affinity_matrix()
        Returns the pariwsie affinity matrix between all elements
    return_network()
        Returns the latent network found by Gower's Method
    return_data_matrix()
        returns the loaded-in data
    subgroup()
        Returns result of Python implementation of Undirected Louvain Method for Clustering Networks
    """
    
    def __init__(self, gowers_scheme='entropy', construction_method='modularity', 
                 epsilon =0.01):
        self.gowers_scheme = gowers_scheme
        self.epsilon = epsilon
        self.construction_method = construction_method
        self.network = None
        
    def return_data_matrix(self):
        """return the loaded-in data
    
        Parameters
        ----------
        None
        
        Returns
        -------
        raw_data : numpy array or list of numpy arrays
            returns the loaded in data in the format the methods of this implementation
            expects
        """
        
        return self.raw_data
    
    def return_network(self):
        """Returns the latent network found by Gower's Method
    
        Parameters
        ----------
        None
        
        Returns
        -------
        network : numpy array
            adjacency matrix of the latent network
        """
        
        return self.network.copy()
            
    def load_data_from_file(self, data, remove_non_distinguishing=False):
        """Read in raw data from a filename or list of file names
    
        Parameters
        ----------
        data : str or list
            filename of the data to read in, specified as a string. Or, in the case of
            multi-mode data a list of strings, where each string is the filename of each
            mode. Data types include .csv or .pkl. Note: null responses
            for categorical or binary data are denoted as '0'.
        remove_non_distinguishing: boolean, optional
            Automatically remove columns from the data where every entry is the same;
            remove columns that are non-distinguishing in terms of the data. Default is
            False.
        
        Creates
        -------
        raw_data : numpy array or list of numpy arrays
            returns the loaded in data in the format the methods of this implementation
            expects
        name_list : pandas series
            list of the names of the entities that will becomes the nodes
            in the latent graph
        """
        
        data_ops = data_operations()
        if type(data) ==list:
            data_type = 'multi-mode'
        else:
            data_type='bipartite'
        
        self.raw_data, self.name_list = data_ops.load_data_from_file(data, data_type=data_type, remove_non_distinguishing=remove_non_distinguishing)
#        self.raw_data = [i[:50,:] for i in self.raw_data]
#        self.name_list = self.name_list[:50]
        
        
    def load_data_from_memory(self, data, remove_non_distinguishing=False):
        """Read in raw data already in memory and convert to the format expected by the methods
    
        Parameters
        ----------
        data : pandas dataframe or list of pandas dataframes
            data to be used by the method, specified as a dataframe, or list of dataframes
            in the case of multi-modal data. Each mode has its own dataframe. Note: null responses
            for categorical or binary data are denoted as '0'.
        remove_non_distinguishing: boolean, optional
            Automatically remove columns from the data where every entry is the same;
            remove columns that are non-distinguishing in terms of the data. Default is
            False.
        
        Creates
        -------
        raw_data : numpy array or list of numpy arrays
            returns the loaded in data in the format the methods of this implementation
            expects
        name_list : pandas series
            list of the names of the entities that will becomes the nodes
            in the latent graph
        """
        
        data_ops = data_operations()
        if type(data) ==list:
            data_type = 'multi-mode'
        else:
            data_type='bipartite'
            
        self.raw_data, self.name_list = data_ops.load_data_from_memory(data, data_type=data_type, remove_non_distinguishing=remove_non_distinguishing)
        
        
    def learn_graph(self):
        """Function call to do Gower's method on the read-in raw data and return the latent graph
    
        Parameters
        ----------
        None
        
        Creates
        -------
        X : numpy array
            the pairwise simialrity matrix from computing Gower's Coefficient of Similarity
            between all of the elements
        
        Returns
        -------
        network : numpy array
            adjacency matrix of the latent network
        """
        
        computation = graph_computation(self.gowers_scheme, self.epsilon, self.construction_method)        
        if self.network == None:
            self.X = computation.compute_similarity(self.raw_data)
            raw_network = computation.compute_graph(self.X)
            self.network = pd.DataFrame(raw_network, index=self.name_list, columns=self.name_list)
            return self.network
        else:
            return self.network
        
    def subgroup(self):
        """Python implementation of Undirected Louvain Method for Clustering Networks
    
        Parameters
        ----------
        None
        
        Returns
        -------
        subgroups : pandas dataframe
            subgroup assignments of all of the nodes, based upon the found latent network
        """
        G = nx.from_pandas_adjacency(self.network)
        subgroups = community.best_partition(G)
        return pd.DataFrame.from_dict(subgroups, orient='index')
        
    def return_affinity_matrix(self):
        """Return the pariwsie affinity matrix between all elements
    
        Parameters
        ----------
        None
        
        Raises
        ------
        Rasises error if learn_graph not called first
        
        Returns
        -------
        data : pandas dataframe
            the pairwise simialrity matrix from computing Gower's Coefficient of Similarity
            between all of the elements.
        
        """
        
        data = pd.DataFrame(data = self.X, index = self.name_list, columns = self.name_list)

        return data
    
    
class data_operations:
    """Helper class to read-in and format data for Gower's Method
    
    Attributes
    ----------
    None
        
    Methods
    -------
    load_data_from_file(data, remove_non_distinguishing=False)
        Read in raw data from a filename or list of file names
    load_data_from_memory(data, remove_non_distinguishing=False)
        Read in raw data already in memory and convert to the format expected by the methods
    """
    
    def __init__(self):
        pass
        
    def _remove_uniform_space(m):
        for col in m:
            if len(m[col].unique()) == 1:
                del m[col]
            
        return m
        
    
    def load_data_from_file(self, path, data_type='bipartite', 
                            remove_non_distinguishing=False):
        if data_type == 'bipartite':
            data = self._load_incidence_from_file(path, remove_non_distinguishing)
            return np.array(data), data.index.to_series()
        elif data_type == 'multi-mode':
            full_set = self._load_incidence_from_file(path[0], remove_non_distinguishing)
            idx = [len(full_set.columns)]
            for file in path[1:]:
                datum = self._load_incidence_from_file(file, remove_non_distinguishing)
                full_set = pd.concat([full_set, datum], axis=1, sort=False)
                idx.append(len(datum.columns)+idx[-1])
                
            name_list = full_set.index.to_series().reset_index(drop=True)
            full_set.fillna(value=0, inplace=True)
            
            m =[full_set.iloc[:,0:idx[0]].values]
            for value in range(1, len(idx)):
                m.append(full_set.iloc[:,idx[value-1]:idx[value]].values)
            return m, name_list
    
    
    def _load_incidence_from_file(self, path, remove_non_distinguishing=False):
        if path[-4:] =='.pkl':
            df = pd.read_pickle(path, encoding="utf-8")
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8", index_col=0)
            except pd.errors.DtypeWarning:
                df = pd.read_csv(path, encoding="utf-8", index_col =0, low_memory=False)
            
        incidence_df = df.copy()
        if incidence_df.dtypes.all() == int or incidence_df.dtypes.all() == float:
            incidence_df.fillna(0, inplace=True)
        else:
            incidence_df.fillna("0", inplace=True)
            
        if np.isin(np.unique(incidence_df.values),[0,1]).all():
            incidence_df.astype(str)
              
        if remove_non_distinguishing:
            incidence_df = self._remove_uniform_space(incidence_df)
            
        return incidence_df
    
    
    def load_data_from_memory(self, data, data_type='bipartite', remove_non_distinguishing=False):
        if data_type == 'bipartite':
            if data.dtypes.all() == int or data.dtypes.all() == float:
                data.fillna(0, inplace=True)
            else:
                data.fillna("0", inplace=True)
                
            if remove_non_distinguishing:
                data = self._remove_uniform_space(data)
            return data, data.index.to_series().reset_index(drop=True)
        elif data_type == 'multi-mode':
            full_set = data[0]
            idx = [len(full_set)]
            for datum in data[1:]:
                if datum.dtypes.all() == int or datum.dtypes.all() == float:
                    datum.fillna(0, inplace=True)
                else:
                    datum.fillna("0", inplace=True)
                    
                if remove_non_distinguishing:
                    datum = self._remove_uniform_space(datum)
                full_set = full_set.merge(datum, how = 'outer', left_index=True, right_index=True)
                idx.append(len(datum)+idx[-1])
                
            name_list = full_set.index.to_series().reset_index(drop=True)
            m =[full_set.iloc[:,0:idx[0]].values]
            for value in range(len(idx[1:])):
                m.append(full_set.iloc[:,idx[value-1]:idx[value]].values)
            
            return (np.array(m), name_list)
     
        
class graph_computation:
    """Core class that handles simialrity calculations and graph learning
    
    Attributes
    ----------
    gowers_scheme : str or list, optional
        The type of weighting scheme to use. Default is entropy. Use 'unweighted'
        for no weights. Pass a list of custom weights for using custom weights;
        note the length of the list must be the same as the number of columns across
        all modes of the data.
    epsilon : float, optional
        specify the threshold for not including edges in the weighted_consensus_graph.
        default is 0.01
    construction_method : str, optional
        the type of graph construction method to use. Current implemented options
        are the modularity k-NN, denoted as 'modularity' which is the default and the
        weighted consensus graph which is denoted as 'WCG'.
        
    Methods
    -------
    compute_similarity(m)
        compute the pairwise simialrity between all entities using Gower's Coefficient
        of Similarity (weighted or unweighted)
    compute_graph(A)
        Learn the graph from the defined construction method on a pairwise similarity
        matrix, A.
    """
    
    def __init__(self, gowers_scheme='entropy', epsilon=0.1, construction_method='modularity'):
        self.gowers_scheme = gowers_scheme
        self.epsilon = epsilon
        self.construction_method = construction_method
    
    
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
        N = A.shape[0]

        if self.construction_method == 'WCG':
            idxs = []
            for i in range(N):
                nonzero_entries = np.nonzero(A[i,:])[0]
                sorted_nonzero_idxs = np.argsort(A[i,nonzero_entries])
                idxs.append(nonzero_entries[sorted_nonzero_idxs])
                
            with mp.Pool(processes=6) as pool:
                graphs = np.array(pool.starmap(self._weighted_consensus_graph, [(int(k), idxs, A) for k in np.floor(np.logspace(0.1,0.9,num=20,base=N))]))
            C = np.sum(graphs, axis=0)
            
            degree = np.sum(C, axis=1)
            D= degree[:, np.newaxis] 
            return np.divide(C, D, out=np.zeros_like(C), where=D!=0)
        else:
            return self._kNN_modularity(A)
        
    
    def _weighted_consensus_graph(self, k, idxs, A):
        N = A.shape[0]
        C = np.zeros((N,N))
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
        
        return C
    
    
    def _kNN_modularity(self, A):
        best_modularity =-1
        best_network = nx.Graph()
        best_k =2
        distances = 1-A
        np.fill_diagonal(distances, np.infty)
    
        for n in range(1, np.int(np.floor(np.log2(distances.shape[0])))):
            k = 2**n
            nodes = np.argpartition(distances, k)[:,:k]
            edges = []
            for i in range(nodes.shape[0]):
                for j in nodes[i,:]:
                    edges.append((i,j))
                    
            network = nx.DiGraph()
            network.add_nodes_from(nodes[:,0])
            network.add_edges_from(edges)
            network = network.to_undirected(reciprocal=False)
            S = network.number_of_nodes()
            p = nx.density(network)
            
            current_clusters = community.best_partition(network)
            random_modularity = (1-2/np.sqrt(S))*(2/(p*S))**(2/3)
            current_modularity = community.modularity(current_clusters, network) - random_modularity

            if current_modularity > best_modularity:
                best_modularity = current_modularity
                best_network = network
                best_k = k

        print("The modularity of the best learned graph was: {}, at a k-value of: {}".format(best_modularity, best_k))
        return nx.to_numpy_array(best_network)
    
        
        
        
        
        
        
        
        
        
        
        