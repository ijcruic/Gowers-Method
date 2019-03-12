# Gowers-Method
Gowers Method for finding latent networks of multi-modal data

## Overview
This project contaisn the code for Gower's Method, which is a method latent networks of multi-modal data. In this context, multi-modal or multi-view data, is data which
has more than one mode of measurement. As a result, we employ Gower's Coefficient of Similarity combined with weighting by network entropy to find latent positions
of points, which we then learna 'best-fit' graph to those points.

Gower's Coefficient of Similarity is given by:
\begin{equation}
S_{ij}=\frac{\sum_{k=1}^{n}w_{ijk}S^{(k)}_{ij}}{\sum_{k=1}^{K}w_{ijk}}\label{sij}
\end{equation}

where $S_{ij}$ is the similarity between datum $i$ and $j$ on a variable, $k$, and $K$ is the total number of variables across all $N$ modes,
 and $w_{ijk}$ is the weight of the similarity between datum $i$ and datum $j$ for variable $k$. $S^{(k)}_{ij}$ is then dually defined as: 

\begin{equation}
S^{(k)}_{ij}:\left\{\begin{matrix} 1, & if (x_{ik}=x_{jk}) \neq \emptyset\\ 0, & otherwise \end{matrix}\right.
\label{sijk_dummy}
\end{equation}

if the variable, $k$, is categorical (to include binary) for node $i$ and $j$'s responses, $x_{ik}$, $x_{jk}$, and: 

\begin{equation}
S^{(k)}_{ij}: \frac{\left | x_{ik}-x_{jk} \right |}{r_{k}}\label{sijk}
\end{equation}

where $r_{k}$ is the range of $x_{k}$, if $k$ is numerical. For each variable, $k$ that is numerical the range is calculated as:

\begin{equation}
r^k = |max(x_k) - min(x_k)|
\end{equation}

The weighting of different modes, $n$, is done by network entropy (which is derived from Quatum or von Neumann Entropy), which is given by:

\begin{equation}
    H^{n} = -\sum^{|V|}_{i=1}\frac{\Tilde{\lambda_i}}{|V|} ln \frac{\Tilde{\lambda_i}}{|V|}
\end{equation}

## Installation

To install the package, you can use pip's built-in functionality  with:
```python
pip install git+https://github.com/ijcruic/Gowers-Method#egg=GowersMethod
```

## Usage
The general usage of the code follows the format of: 
- importing the package
- creating a latent_graph object
- adding data to that latent graph object
- And, finally, fitting a graph to the data in the latent graph object

The following code details an example

```python
import GowersMethod as GM

mode_files = ['Mode_1.csv', 'Mode_2.csv', 'Mode_3.csv']

if __name__ == "__main__":
    lg = GM.latent_graph(epsilon = 0)
    lg.load_data_from_file(mode_files)
    network = lg.learn_graph()
```
### Input


### Output


## References
* Ruan, J.: A fully automated method for discovering community structures in high
dimensional data. In: 2009 Ninth IEEE International Conference on Data Mining.
pp. 968{973 (Dec 2009). https://doi.org/10.1109/ICDM.2009.141

* Guimera R, Sales-Pardo M, Amaral LN. Modularity from fluctuations in random graphs and
complex networks. Physical Review E. 2004; 70:025101.
