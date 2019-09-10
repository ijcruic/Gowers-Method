# -*- coding: utf-8 -*-
"""
Package for implmenting Gower's method for finding latent networks in multi-modal data

author: Iain Cruickshank
license: MIT version 3
acknowledgements: Work supported by National Science Foundation Graduate Research Fellowship (DGE 1745016).   
"""

import pandas as pd, networkx as nx, community
from Gowers_Method.data_operations import data_operations
from Gowers_Method.graph_computation import graph_computation

class latent_graph:
    """Finds a latent graph of data using Gower's method and specified graph constructor
    
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
                 WCG_epsilon =0.1, knn_symmetrize= False, network_enhancement=True, enhancement_iterations=100, 
                 enhancement_alpha=0.9, enhancement_nearest_neighbors='sqrt'):
        self.gowers_scheme = gowers_scheme
        self.epsilon = WCG_epsilon
        self.symmetrize = knn_symmetrize
        self.construction_method = construction_method
        self.network_enhancement = network_enhancement
        self.T = enhancement_iterations
        self.alpha = enhancement_alpha
        self.nearest_neighbors = enhancement_nearest_neighbors
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
        
        computation = graph_computation(self.gowers_scheme, self.construction_method, 
                                        self.epsilon, self.symmetrize, 
                                        self.network_enhancement, self.T, self.alpha, 
                                        self.nearest_neighbors)        
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