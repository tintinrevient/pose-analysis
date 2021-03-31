import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import os

outfile = os.path.join('pix', 'dendrogram.png')

POSE_DATA_CSV = os.path.join('output', 'joint_angles.csv')
pose_data = pd.read_csv(POSE_DATA_CSV, index_col=0)

df = pose_data
columns = df.columns

# filter for one artist
df = df.loc[df.index.str.contains('Michelangelo'), columns]
print(df)

Z = linkage(df, method='complete')
print(Z.shape)

# plot the dendrogram

# for all artists
# fig, ax = plt.subplots(figsize=(30, 5))

# dimension for one artist
fig, ax = plt.subplots(figsize=(8, 5))

dendrogram(Z, labels=list(df.index), color_threshold=0)
plt.xticks(rotation=90)
ax.set_ylabel('distance')

# save the plot
plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=227)

# show the plot
# plt.tight_layout()
# plt.show()

memb = fcluster(Z, 4, criterion='maxclust')
memb = pd.Series(memb, index=df.index)
for key, item in memb.groupby(memb):
    print(f"{key} : {', '.join(item.index)}")