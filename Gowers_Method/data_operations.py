# -*- coding: utf-8 -*-
"""
Package for implmenting Gower's method for finding latent networks in multi-modal data

author: Iain Cruickshank
license: MIT version 3
acknowledgements: Work supported by National Science Foundation Graduate Research Fellowship (DGE 1745016).   
"""

import pandas as pd, numpy as np

class data_operations:
    
    def __init__(self):
        pass
        
    def _remove_uniform_space(m):
        for col in m:
            if len(m[col].unique()) == 1:
                del m[col]
            
        return m
        
    
    def load_data_from_file(self, path, data_type='bipartite', 
                            remove_non_distinguishing=False):
        """Load in data from a list of paths
    
        Parameters
        ----------
        path : the list of paths to the files containing the different modes of the data
        
        Returns
        -------
        m : 
        """
        
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
