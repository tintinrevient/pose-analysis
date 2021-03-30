import math
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import multivariate_normal

import prince

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import from_levels_and_colors
import seaborn as sns

import os

# import the data

SP500_DATA_CSV = os.path.join('dataset', 'sp500_data.csv.gz')
sp500_data = pd.read_csv(SP500_DATA_CSV, index_col=0)

columns = ['AAPL', 'AMZN', 'AXP', 'COP', 'COST', 'CSCO', 'CVX', 'GOOGL', 'HD',
         'INTC', 'JPM', 'MSFT', 'SLB', 'TGT', 'USB', 'WFC', 'WMT', 'XOM']

df = sp500_data.loc[sp500_data.index >= '2011-01-01', columns].transpose()

Z = linkage(df, method='complete')
print(Z.shape)

# plot the dendrogram

fig, ax = plt.subplots(figsize=(5, 5))
dendrogram(Z, labels=list(df.index), color_threshold=0)
plt.xticks(rotation=90)
ax.set_ylabel('distance')

plt.tight_layout()
plt.show()

memb = fcluster(Z, 4, criterion='maxclust')
memb = pd.Series(memb, index=df.index)
for key, item in memb.groupby(memb):
    print(f"{key} : {', '.join(item.index)}")

# measure the dissimilarity

df = sp500_data.loc[sp500_data.index >= '2011-01-01', ['XOM', 'CVX']]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

for i, method in enumerate(['single', 'average', 'complete', 'ward']):

    ax = axes[i // 2, i % 2]
    Z = linkage(df, method=method)
    colors = [f'C{c+1}' for c in fcluster(Z, 4, criterion='maxclust')]

    ax = sns.scatterplot(data=df, x='XOM', y='CVX', hue=colors, style=colors, ax=ax, legend=False)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(method)

plt.tight_layout()
plt.show()