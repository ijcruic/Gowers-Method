# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:05:52 2019

@author: icruicks
"""
import numpy as np

class thermodynamic_measures:
    """Toolkit for computing various graph thermodynamic measures for a given graph
    
    Attributes
    ----------
    None
        
    Methods
    -------
    approx_entropy : compute the approximate entropy of a graph. This is much faster 
        than computing the exact entropy of a graph, but assumes the graph is unweighted
        and undirected. So, a weighted graph will be binarized before calculation
    exact_entropy : compute the exact entropy of a graph
    internal_energy : compute the internal energy of the graph. This measure is 
        related to the number of edges present in the graph
    temperature : takes two graphs as input and compares their difference in temperature.
        Temperature is a ratio between the differences in entropy of the graphs 
        versus a differenc of internal energy of the graphs.
    """
        
    
    def approx_entropy(G):
        N = G.shape[0]
        degree = np.count_nonzero(G,axis=1)
        degree_match = np.multiply(G>0, np.outer(degree, degree)).astype(float)
        inversion_term = np.ones_like(degree_match)
        degree_term = np.divide(inversion_term, degree_match, 
                                out=np.zeros_like(inversion_term), where=degree_match!=0)
        return 1- 1/N - (1/(N**2))*np.sum(degree_term)
    
    def exact_entropy(G):
        N= G.shape[0]
        D= np.count_nonzero(G, axis=1)*np.eye(N)
        D_inv = np.nan_to_num(np.power(np.count_nonzero(G,axis=1),-0.5)*np.eye(N))
        L = np.nan_to_num(np.matmul(D_inv,np.matmul(D-G, D_inv)))
        eigenvalues = np.real(np.linalg.eig(L)[0])
        eigenvalues[eigenvalues<1e-9] =0 
        eigenvalues = eigenvalues[eigenvalues.nonzero()]
        return -1* np.sum(np.nan_to_num(np.multiply(eigenvalues/N, np.log(eigenvalues/N))))
    
    def internal_energy(G):
        N = G.shape[0]
        degree = np.count_nonzero(G, axis=1)
        return np.sum(np.multiply(N, degree))
    
    def temperature(G_1,G_2):
        T_inv = (thermodynamic_metrics.approx_entropy(G_2) - thermodynamic_metrics.approx_entropy(G_1))/ \
        (thermodynamic_metrics.internal_energy(G_2) - thermodynamic_metrics.internal_energy(G_1))
        
        return 1/T_inv
        