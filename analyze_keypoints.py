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
        dx1 = point1[0] - center[0]
        dx2 = point2[0] - center[0]
        dy1 = point1[1] - center[1]
        dy2 = point2[1] - center[1]
        m1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
        m2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

        rad = np.arccos((dx1 * dx2 + dy1 * dy2) / (m1 * m2))
        deg = rad * 180.0 / np.pi

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
        reference_point = keypoints['MidHip'] + [0, -100, 0]
        rad, deg = calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

        # rotate the joints around keypoints['MidHip']
        if keypoints['Neck'][0] > keypoints['MidHip'][0]:
            rad = -rad

        for key, value in keypoints.items():
            rotated_keypoints[key] = rotate(value, keypoints['MidHip'], rad)

        ######################
        # Angles of 3 Joints #
        ######################
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

        ##################
        # Normalize pose #
        ##################
        image_norm = np.zeros(data['dimension'], np.uint8)

        for index, pair in enumerate(joint_pairs):

            point1 = keypoints.get(pair[0])
            point2 = keypoints.get(pair[1])

            rotated_point1 = rotated_keypoints.get(pair[0])
            rotated_point2 = rotated_keypoints.get(pair[1])

            if point1[2] != 0 and point2[2] != 0:

                # draw original keypoints in yellow
                point1_xy = (int(point1[0]), int(point1[1]))
                point2_xy = (int(point2[0]), int(point2[1]))
                cv2.line(image, point1_xy, point2_xy, color=(0, 255, 255), thickness=5)

                # draw rotated keypoints in magenta
                rotated_point1_xy = (rotated_point1[0], rotated_point1[1])
                rotated_point2_xy = (rotated_point2[0], rotated_point2[1])
                cv2.line(image, rotated_point1_xy, rotated_point2_xy, color=(255, 0, 255), thickness=5)

                # draw rotated keypoints in normalized image
                cv2.line(image_norm, rotated_point1_xy, rotated_point2_xy, color=(255, 0, 255), thickness=5)

        image_norm_bbox = clip_bbox(image_norm, rotated_keypoints, data['dimension'])
        image_resize = cv2.resize(image_norm_bbox, (300, 300), interpolation=cv2.INTER_AREA)
        norm_fname = image_fname.replace('_rendered', '_' + str(person_index) + '_norm')
        cv2.imwrite(norm_fname, image_resize)
        print('output', norm_fname)

    if show:
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_dict, output_index


if __name__ == '__main__':

    # python analyze_keypoints.py --input output/data/

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
