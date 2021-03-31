import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import os


POSE_DATA_CSV = os.path.join('output', 'joint_angles.csv')
pose_data = pd.read_csv(POSE_DATA_CSV, index_col=0)

df = pose_data
print(df)

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