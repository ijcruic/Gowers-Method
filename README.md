# Gowers-Method
Gowers Method for finding latent networks of multi-modal data

## Overview
This project contains the code for Gower's Method, which is a method for finding latent networks of multi-modal data. In this context, multi-modal or multi-view data, is data which
has more than one mode of measurement. As a result, we employ Gower's Coefficient of Similarity combined with weighting by network entropy to find latent positions
of points, which we then learn a 'best-fit' graph to those points.

Options available in this package include: 
- Use it on multi-modal or one-mode data. 
- Graph learning methods are: modularity k-NN (default), and Weighted Consensus Graph
- Weighting schemes are: network entropy (default), none, and user-supplied

## Installation

To install the package, you can use pip's built-in functionality  with:
```python
pip install git+https://github.com/ijcruic/Gowers-Method#egg=Gowers_Method
```

## Usage
The general usage of the code follows the format of: 
- importing the package
- creating a latent_graph object
- adding data to that latent graph object
- And, finally, fitting a graph to the data in the latent graph object

The following code details an example

```python
from Gowers_Method import latent_graph

mode_files = ['Mode_1.csv', 'Mode_2.csv', 'Mode_3.csv']

if __name__ == "__main__":
    lg = latent_graph()
    lg.load_data_from_file(mode_files)
    network = lg.learn_graph()
```
### Input
There are some inputs to the method to be aware of, including the data and the weighting scheme. Data can either be submitted as a list of files,
where each file is a mode of the data or a list of Pandas data frames, where each data frame is a mode of the data. The weighting scheme can be 
unspecified, in which case it will be network entropy. It can also be specified as 'unweighted' in which case it will be unweighted (the weight vector will
be all ones). If you wish to have a user specified weight scheme, you must a pass a list of length number of variables, where each entry is the numerical
weight desired for that variable. Finally, graph construction can be  unspecified, in modularity k-NN will be used, or specified as 'WCG' to learn a 
Weighted Consensus Graph.

### Output
Output of the method will be the graph adjacency in a Pandas data frame, where the index and columns are the data names. 

## References
* Campedelli G.M., Cruickshank I., Carley K.M. (2019) Detecting Latent Terrorist Communities Testing a Gower’s Similarity-Based Clustering Algorithm for Multi-partite Networks. In: Aiello L., Cherifi C., Cherifi H., Lambiotte R., Lió P., Rocha L. (eds) Complex Networks and Their Applications VII. COMPLEX NETWORKS 2018. Studies in Computational Intelligence, vol 812. Springer, Cham
