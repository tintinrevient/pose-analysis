import numpy as np
import os
import cv2
from scipy import ndimage
import argparse
from pathlib import Path
import pandas as pd


# 1. Body 25 keypoints
joint_ids = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background'
]

joint_pairs = [
    # ('REar', 'REye'), ('LEar', 'LEye'), ('REye', 'Nose'), ('LEye', 'Nose'),
    ('Nose', 'Neck'), ('Neck', 'MidHip'),
    ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('MidHip', 'RHip'), ('MidHip', 'LHip'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle')
]

joint_triples = [
    ('Nose', 'Neck', 'MidHip'),
    ('RShoulder','Neck', 'MidHip'), ('LShoulder', 'Neck', 'MidHip'),
    ('RElbow', 'RShoulder','Neck'), ('LElbow', 'LShoulder', 'Neck'),
    ('RWrist', 'RElbow', 'RShoulder'), ('LWrist', 'LElbow', 'LShoulder'),
    ('RHip', 'MidHip', 'Neck'), ('LHip', 'MidHip', 'Neck'),
    ('RKnee', 'RHip', 'MidHip'), ('LKnee', 'LHip', 'MidHip'),
    ('RAnkle', 'RKnee', 'RHip'), ('LAnkle', 'LKnee', 'LHip')
]

# 'zero' and 'nan' will result in errors in hierarchical clustering
minimum_positive_above_zero = np.nextafter(0, 1)


def euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def calc_angle(point1, center, point2):

    try:
        a = np.array(point1)[0:2] - np.array(center)[0:2]
        b = np.array(point2)[0:2] - np.array(center)[0:2]

        cos_theta = np.dot(a, b)
        sin_theta = np.cross(a, b)

        rad = np.arctan2(sin_theta, cos_theta)
        deg = np.rad2deg(rad)

        if np.isnan(rad):
            return minimum_positive_above_zero, minimum_positive_above_zero

        return rad, deg

    except:
        return minimum_positive_above_zero, minimum_positive_above_zero


def rotate(point, center, rad):

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0];
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1];

    return [int(x), int(y), point[2]]


def is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    # filter the main keypoints by score = 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] == 0]

    if len(filtered_keypoints) != 0:
        return False
    else:
        return True


def clip_bbox(image, keypoints, dimension):
    '''
    for keypoints of one person
    '''

    min_x = dimension[1]
    max_x = 0
    min_y = dimension[0]
    max_y = 0

    for key, value in keypoints.items():
        x, y, score = value

        if score == 0.0:
            continue

        if x < min_x and x >=0:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y and y >=0:
            min_y = y
        if y > max_y:
            max_y = y

    x = int(min_x)
    y = int(min_y)
    w = int(max_x - min_x)
    h = int(max_y - min_y)

    image_bbox = image[y:y + h, x:x + w]

    return image_bbox


def calc_joint_angle(output_dict, keypoints):
    '''
    for keypoints of one person
    '''

    for index, triple in enumerate(joint_triples):

        point1 = keypoints.get(triple[0])
        center = keypoints.get(triple[1])
        point2 = keypoints.get(triple[2])

        col_name = '{}_{}_{}'.format(triple[0], triple[1], triple[2])

        if col_name not in output_dict:
            output_dict[col_name] = []

        if point1[2] != 0 and center[2] != 0 and point2[2] != 0:
            rad, deg = calc_angle(point1=point1, center=center, point2=point2)
            output_dict[col_name].append(rad)
        else:
            output_dict[col_name].append(minimum_positive_above_zero)


def xy_tuple(arr):
    return tuple(arr[0:2])


def norm_pose(rotated_keypoints, show):
    '''
    for keypoints of one person
    '''

    # white background image
    image = np.empty((300, 300, 3), np.uint8)
    image.fill(255)

    # drawing settings
    line_color = (0, 0, 255) # bgr
    line_thickness = 3

    # normalized joint locations (x, y, score)
    neck_xy = (150, 100, 1)
    midhip_xy = (150, 170, 1) # length of body = 70 -> (170 - 100)
    upper_xy = (150, 70, 1) # length of limbs + nose = 30 -> (100 - 70)
    lower_xy = (150, 140, 1) # length of limbs = 30 -> (170 - 140)


    # Neck to MidHip as base
    cv2.line(image, xy_tuple(neck_xy), xy_tuple(midhip_xy), color=line_color, thickness=line_thickness)

    # Nose to Neck
    if rotated_keypoints.get('Nose')[2] != 0:
        rad, deg = calc_angle(np.array(rotated_keypoints.get('Neck')) + np.array([0, -50, 0]), rotated_keypoints.get('Neck'), rotated_keypoints.get('Nose'))
        nose_xy = rotate(upper_xy, neck_xy, rad)
        cv2.line(image, xy_tuple(nose_xy), xy_tuple(neck_xy), color=line_color, thickness=line_thickness)

    # RIGHT
    # RShoulder to Neck
    rad, deg = calc_angle(np.array(rotated_keypoints.get('Neck')) + np.array([0, -50, 0]), rotated_keypoints.get('Neck'), rotated_keypoints.get('RShoulder'))
    rsho_xy = rotate(upper_xy, neck_xy, rad)
    cv2.line(image, xy_tuple(rsho_xy), xy_tuple(neck_xy), color=line_color, thickness=line_thickness)

    # RElbow to RShoulder
    if rotated_keypoints.get('RElbow')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('Neck'), rotated_keypoints.get('RShoulder'), rotated_keypoints.get('RElbow'))
        relb_xy = rotate(neck_xy, rsho_xy, rad)
        cv2.line(image, xy_tuple(relb_xy), xy_tuple(rsho_xy), color=line_color, thickness=line_thickness)

    # RWrist to RElbow
    if rotated_keypoints.get('RElbow')[2] != 0 and rotated_keypoints.get('RWrist')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('RShoulder'), rotated_keypoints.get('RElbow'), rotated_keypoints.get('RWrist'))
        rwrist_xy = rotate(rsho_xy, relb_xy, rad)
        cv2.line(image, xy_tuple(rwrist_xy), xy_tuple(relb_xy), color=line_color, thickness=line_thickness)

    # RHip to MidHip
    rad, deg = calc_angle(rotated_keypoints.get('Neck'), rotated_keypoints.get('MidHip'), rotated_keypoints.get('RHip'))
    rhip_xy = rotate(lower_xy, midhip_xy, rad)
    cv2.line(image, xy_tuple(rhip_xy), xy_tuple(midhip_xy), color=line_color, thickness=line_thickness)

    # RKnee to RHip
    if rotated_keypoints.get('RKnee')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('MidHip'), rotated_keypoints.get('RHip'), rotated_keypoints.get('RKnee'))
        rknee_xy = rotate(midhip_xy, rhip_xy, rad)
        cv2.line(image, xy_tuple(rknee_xy), xy_tuple(rhip_xy), color=line_color, thickness=line_thickness)

    # RAnkle to RKnee
    if rotated_keypoints.get('RKnee')[2] != 0 and rotated_keypoints.get('RAnkle')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('RHip'), rotated_keypoints.get('RKnee'), rotated_keypoints.get('RAnkle'))
        rankle_xy = rotate(rhip_xy, rknee_xy, rad)
        cv2.line(image, xy_tuple(rankle_xy), xy_tuple(rknee_xy), color=line_color, thickness=line_thickness)

    # LEFT
    # LShoulder to Neck
    rad, deg = calc_angle(np.array(rotated_keypoints.get('Neck')) + np.array([0, -50, 0]), rotated_keypoints.get('Neck'), rotated_keypoints.get('LShoulder'))
    lsho_xy = rotate(upper_xy, neck_xy, rad)
    cv2.line(image, xy_tuple(lsho_xy), xy_tuple(neck_xy), color=line_color, thickness=line_thickness)

    # LElbow to LShoulder
    if rotated_keypoints.get('LElbow')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('Neck'), rotated_keypoints.get('LShoulder'), rotated_keypoints.get('LElbow'))
        lelb_xy = rotate(neck_xy, lsho_xy, rad)
        cv2.line(image, xy_tuple(lelb_xy), xy_tuple(lsho_xy), color=line_color, thickness=line_thickness)

    # LWrist to LElbow
    if rotated_keypoints.get('LElbow')[2] != 0 and rotated_keypoints.get('LWrist')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('LShoulder'), rotated_keypoints.get('LElbow'), rotated_keypoints.get('LWrist'))
        lwrist_xy = rotate(lsho_xy, lelb_xy, rad)
        cv2.line(image, xy_tuple(lwrist_xy), xy_tuple(lelb_xy), color=line_color, thickness=line_thickness)

    # LHip to MidHip
    rad, deg = calc_angle(rotated_keypoints.get('Neck'), rotated_keypoints.get('MidHip'), rotated_keypoints.get('LHip'))
    lhip_xy = rotate(lower_xy, midhip_xy, rad)
    cv2.line(image, xy_tuple(lhip_xy), xy_tuple(midhip_xy), color=line_color, thickness=line_thickness)

    # LKnee to LHip
    if rotated_keypoints.get('LKnee')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('MidHip'), rotated_keypoints.get('LHip'), rotated_keypoints.get('LKnee'))
        lknee_xy = rotate(midhip_xy, lhip_xy, rad)
        cv2.line(image, xy_tuple(lknee_xy), xy_tuple(lhip_xy), color=line_color, thickness=line_thickness)

    # LAnkle to LKnee
    if rotated_keypoints.get('LKnee')[2] != 0 and rotated_keypoints.get('LAnkle')[2] != 0:
        rad, deg = calc_angle(rotated_keypoints.get('LHip'), rotated_keypoints.get('LKnee'), rotated_keypoints.get('LAnkle'))
        lankle_xy = rotate(lhip_xy, lknee_xy, rad)
        cv2.line(image, xy_tuple(lankle_xy), xy_tuple(lknee_xy), color=line_color, thickness=line_thickness)

    if show:
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def visualize_rotated_pose(image, keypoints, rotated_keypoints):

    for index, pair in enumerate(joint_pairs):

        point1 = keypoints.get(pair[0])
        point2 = keypoints.get(pair[1])

        rotated_point1 = rotated_keypoints.get(pair[0])
        rotated_point2 = rotated_keypoints.get(pair[1])

        if point1[2] != 0 and point2[2] != 0:

            # draw original keypoints in Yellow
            point1_xy = (int(point1[0]), int(point1[1]))
            point2_xy = (int(point2[0]), int(point2[1]))
            cv2.line(image, point1_xy, point2_xy, color=(0, 255, 255), thickness=5)

            # draw rotated keypoints in Red
            rotated_point1_xy = (rotated_point1[0], rotated_point1[1])
            rotated_point2_xy = (rotated_point2[0], rotated_point2[1])
            cv2.line(image, rotated_point1_xy, rotated_point2_xy, color=(0, 0, 255), thickness=5)

    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_keypoints(infile, output_dict={}, output_index=[], show=False):

    data = np.load(infile, allow_pickle='TRUE').item()

    print('input:', infile)
    print('number of people:', data['keypoints'].shape[0])

    person_index = 0
    index_fname = '{}_{}'.format(infile.split('/')[3], infile[infile.rfind('/')+1:infile.rfind('_')])

    image_fname = infile.replace('/data/', '/pix/').replace('_keypoints.npy', '_rendered.png')
    image = cv2.imread(image_fname)

    # iterate through all people
    for keypoints in data['keypoints']:

        # process one person!!!

        # create keypoints + rotated_keypoints
        keypoints = dict(zip(joint_ids, keypoints))
        rotated_keypoints = {}

        # if not valid, skip to the next person
        if not is_valid(keypoints=keypoints):
            continue

        # generate person index
        person_index += 1
        output_index.append('{}_{}'.format(index_fname, person_index))

        ######################
        # Angles of 3 Joints #
        ######################
        # To generate the dendrogram!!!
        calc_joint_angle(output_dict, keypoints)

        ################
        # Bounding Box #
        ################
        image_bbox = clip_bbox(image, keypoints, data['dimension'])

        person_fname = image_fname.replace('_rendered', '_' + str(person_index))
        cv2.imwrite(person_fname, image_bbox)
        print('output', person_fname)

        ############
        # Rotation #
        ############
        reference_point = np.array(keypoints['MidHip']) + np.array([0, -100, 0])
        rad, deg = calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

        for key, value in keypoints.items():
            rotated_keypoints[key] = rotate(value, keypoints['MidHip'], rad)

        ##################
        # Normalize pose #
        ##################
        image_norm = norm_pose(rotated_keypoints, show=show)
        norm_fname = image_fname.replace('_rendered', '_norm_' + str(person_index))
        cv2.imwrite(norm_fname, image_norm)
        print('output', norm_fname)

        # Check if the pose is right!!!
        if show:
            visualize_rotated_pose(image=image, keypoints=keypoints, rotated_keypoints=rotated_keypoints)

    return output_dict, output_index


if __name__ == '__main__':

    # python analyze_keypoints.py --input output/data/

    # Felix\ Vallotton/7043_keypoints.npy
    # Paul\ Delvaux/81511_keypoints.npy
    # Paul\ Delvaux/74433_keypoints.npy

    parser = argparse.ArgumentParser(description='Extract the angles of keypoints')
    parser.add_argument("--input", help="a directory or a single npy keypoints data")
    args = parser.parse_args()

    output_dict = {}
    output_index = []

    if os.path.isfile(args.input):
        load_keypoints(infile=args.input, show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.npy'):
            output_dict, output_index = load_keypoints(infile=str(path), output_dict=output_dict, output_index=output_index)

        df = pd.DataFrame(data=output_dict, index=output_index)
        df.to_csv(os.path.join('output', 'joint_angles.csv'), index=True)
