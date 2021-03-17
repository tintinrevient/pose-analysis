from scipy.io import loadmat
import os
import numpy as np
from math import isnan
import cv2
import matplotlib.pyplot as plt

flic_dir = os.path.join('dataset', 'FLIC')
img_dir = os.path.join(flic_dir, 'images')
mat_dir = os.path.join(flic_dir, 'examples.mat')

# input the image idx = [0, 5003)
img_idx = 500

examples = loadmat(mat_dir)
examples = examples['examples'][0]
print('Total examples:', len(examples))

# image filename
fname = examples[img_idx]['filepath'][0]
print('Image:', fname)

# 1. keypoints
joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri',
             'lhip', 'lkne', 'lank', 'rhip', 'rkne', 'rank',
             'leye', 'reye', 'lear', 'rear', 'nose',
             'msho', 'mhip', 'mear', 'mtorso',
             'mluarm', 'mruarm', 'mllarm', 'mrlarm',
             'mluleg', 'mruleg', 'mllleg', 'mrlleg']

joints = examples[img_idx]['coords'].T
joints = dict(zip(joint_ids, joints))

def get_aggregated_joints(joints):

    # head
    head = (joints['leye'], joints['reye'], joints['nose'])
    head = np.vstack(head)
    del joints['leye']
    del joints['reye']
    del joints['nose']
    joints['head'] = head.tolist()

    # left arm
    larm = (joints['lsho'], joints['lelb'], joints['lwri'])
    larm = np.vstack(larm)
    del joints['lsho']
    del joints['lelb']
    del joints['lwri']
    joints['larm'] = larm.tolist()

    # right arm
    rarm = (joints['rsho'], joints['relb'], joints['rwri'])
    rarm = np.vstack(rarm)
    del joints['rsho']
    del joints['relb']
    del joints['rwri']
    joints['rarm'] = rarm.tolist()

    # hip
    hip = (joints['lhip'], joints['rhip'])
    hip = np.vstack(hip)
    del joints['lhip']
    del joints['rhip']
    joints['hip'] = hip.tolist()

    # clean the list with nan values
    joints = {k: v for k, v in joints.items() if not isnan(np.sum(v))}

    return joints

aggregated_joints = get_aggregated_joints(joints)

# 2. torso bbo
torso_bbox = examples[img_idx]['torsobox'][0]
print('Torso bbox:', torso_bbox)
torso_x1, torso_y1, torso_x2, torso_y2 = torso_bbox[0], torso_bbox[1], torso_bbox[2], torso_bbox[3]

# show the input image
input_img= cv2.imread(os.path.join(img_dir, fname))
plt.imshow(input_img[:,:,::-1])

# 1. keypoints: draw the points
colors = {'head': 'cyan',
          'larm': 'lime',
          'rarm': 'magenta',
          'hip': 'blue'}

for key, coords_list in aggregated_joints.items():
    for coords in coords_list:
        plt.scatter(coords[0], coords[1], s=5, marker='o', color=colors[key])

# head keypoints
head_x1 = aggregated_joints['head'][0][0]
head_y1 = aggregated_joints['head'][0][1]
head_x2 = aggregated_joints['head'][1][0]
head_y2 = aggregated_joints['head'][1][1]
head_x3 = aggregated_joints['head'][2][0]
head_y3 = aggregated_joints['head'][2][1]
plt.plot([head_x1, head_x2], [head_y1, head_y2], color=colors['head'], linewidth=2)
plt.plot([head_x1, head_x3], [head_y1, head_y3], color=colors['head'], linewidth=2)
plt.plot([head_x2, head_x3], [head_y2, head_y3], color=colors['head'], linewidth=2)

# left arm keypoints
larm_x1 = aggregated_joints['larm'][0][0]
larm_y1 = aggregated_joints['larm'][0][1]
larm_x2 = aggregated_joints['larm'][1][0]
larm_y2 = aggregated_joints['larm'][1][1]
larm_x3 = aggregated_joints['larm'][2][0]
larm_y3 = aggregated_joints['larm'][2][1]
plt.plot([larm_x1, larm_x2], [larm_y1, larm_y2], color=colors['larm'], linewidth=2)
plt.plot([larm_x2, larm_x3], [larm_y2, larm_y3], color=colors['larm'], linewidth=2)

# right arm keypoints
rarm_x1 = aggregated_joints['rarm'][0][0]
rarm_y1 = aggregated_joints['rarm'][0][1]
rarm_x2 = aggregated_joints['rarm'][1][0]
rarm_y2 = aggregated_joints['rarm'][1][1]
rarm_x3 = aggregated_joints['rarm'][2][0]
rarm_y3 = aggregated_joints['rarm'][2][1]
plt.plot([rarm_x1, rarm_x2], [rarm_y1, rarm_y2], color=colors['rarm'], linewidth=2)
plt.plot([rarm_x2, rarm_x3], [rarm_y2, rarm_y3], color=colors['rarm'], linewidth=2)

# hip keypoints
hip_x1 = aggregated_joints['hip'][0][0]
hip_y1 = aggregated_joints['hip'][0][1]
hip_x2 = aggregated_joints['hip'][1][0]
hip_y2 = aggregated_joints['hip'][1][1]
plt.plot([hip_x1, hip_x2], [hip_y1, hip_y2], color=colors['hip'], linewidth=2)

# 2. torso bbox: draw a line from [x1, y1] to [x2, y1], etc.
plt.plot([torso_x1, torso_x2], [torso_y1, torso_y1], linestyle='dashed', color='lightgray', linewidth=1)
plt.plot([torso_x1, torso_x1], [torso_y1, torso_y2], linestyle='dashed', color="lightgray", linewidth=1)
plt.plot([torso_x2, torso_x2], [torso_y2, torso_y1], linestyle='dashed', color="lightgray", linewidth=1)
plt.plot([torso_x2, torso_x1], [torso_y2, torso_y2], linestyle='dashed', color="lightgray", linewidth=1)

plt.axis('off')
plt.show()