import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.offsetbox import OffsetImage,AnnotationBbox

# input setting
num_cluster = 5
category = 'classical'
artist = 'Michelangelo'
size = 10

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
    fig, ax = plt.subplots(figsize=(size, 5))
else:
    fig, ax = plt.subplots(figsize=(30, 5))


dendrogram(Z, labels=list(df.index), color_threshold=0)

# xticks
plt.xticks(rotation=90)
# ylabel
ax.set_ylabel('distance')

# show the corresponding images on xticks
def get_flag(name):
    name_list = name.split('_')
    artist = name_list[0]
    fname = '%s_norm_%s.png' % (name_list[1], name_list[2])
    path = "output/pix/%s/%s/%s" % (category, artist, fname)
    im = plt.imread(path)
    return im

def offset_image(xtick, label, ax):
    name = label.get_text()
    img = get_flag(name)
    im = OffsetImage(img, zoom=0.2)
    im.image.axes = ax

    ab = AnnotationBbox(im, (xtick, 0),  xybox=(0., -20.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)

xticks = list(ax.get_xticks())
xticklabels = list(ax.get_xticklabels())

for xtick, label in zip(xticks, xticklabels):
    offset_image(xtick, label, ax)

# don't show axis
plt.axis('off')

# save the plot
plt.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=227)

# show the plot
# plt.tight_layout()
# plt.show()

memb = fcluster(Z, num_cluster, criterion='maxclust')
memb = pd.Series(memb, index=df.index)
for key, item in memb.groupby(memb):
    print(f"{key} : {', '.join(item.index)}")