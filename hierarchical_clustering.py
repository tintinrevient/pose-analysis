import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import os

num_cluster = 5
artist = 'Michelangelo'

if artist:
    outfile = os.path.join('pix', 'dendrogram{}.png'.format('_' + artist))
else:
    outfile = os.path.join('pix', 'dendrogram.png')

POSE_DATA_CSV = os.path.join('output', 'joint_angles.csv')
pose_data = pd.read_csv(POSE_DATA_CSV, index_col=0)

df = pose_data
columns = df.columns

# filter for one artist
if artist:
    df = df.loc[df.index.str.contains(artist), columns]

print(df)

Z = linkage(df, method='complete')
print(Z.shape)

# plot the dendrogram

if artist:
    fig, ax = plt.subplots(figsize=(10, 5))
else:
    fig, ax = plt.subplots(figsize=(30, 5))


dendrogram(Z, labels=list(df.index), color_threshold=0)
plt.xticks(rotation=90)
ax.set_ylabel('distance')

# save the plot
plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=227)

# show the plot
# plt.tight_layout()
# plt.show()

memb = fcluster(Z, num_cluster, criterion='maxclust')
memb = pd.Series(memb, index=df.index)
for key, item in memb.groupby(memb):
    print(f"{key} : {', '.join(item.index)}")