import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np

POSE_DATA_CSV = os.path.join('output', 'joint_angles.csv')
pose_data = pd.read_csv(POSE_DATA_CSV, index_col=0)

df = pose_data.transpose()
columns = df.columns
print(df)

kmeans = KMeans(n_clusters=10).fit(df)

from collections import Counter
print(Counter(kmeans.labels_))

centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)

f, axes = plt.subplots(5, 1, figsize=(5, 6), sharex=True)
for i, ax in enumerate(axes):
    center = centers.loc[i, :]
    maxPC = 1.01 * np.max(np.max(np.abs(center)))
    colors = ['C0' if l > 0 else 'C1' for l in center]
    ax.axhline(color='#888888')
    center.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'Cluster {i + 1}')
    ax.set_ylim(-maxPC, maxPC)

plt.tight_layout()
plt.show()