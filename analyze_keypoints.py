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


def euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def calc_angle(point1, center, point2):

    minimum_positive_above_zero = np.nextafter(0, 1)

    try:
        dx1 = point1[0] - center[0]
        dx2 = point2[0] - center[0]
        dy1 = point1[1] - center[1]
        dy2 = point2[1] - center[1]
        m1 = np.sqrt(dx1 ** 2 + dy1 ** 2);
        m2 = np.sqrt(dx2 ** 2 + dy2 ** 2);

        rad = np.arccos((dx1 * dx2 + dy1 * dy2) / (m1 * m2))
        deg = rad * 180.0 / np.pi;

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


def load_keypoints(infile, output_dict={}, output_index=[], show=False):

    data = np.load(infile, allow_pickle='TRUE').item()

    print('dimension:', data['dimension'])
    print('number of people:', data['keypoints'].shape[0])

    fname = infile[infile.rfind('/') + 1:infile.rfind('_')]
    person_index = 0

    if show:
        image = cv2.imread(os.path.join('test', 'pix', '{}.jpg'.format(fname)))

    # iterate through the keypoints of all the people
    for keypoints in data['keypoints']:

        # process the keypoints of one person
        keypoints = dict(zip(joint_ids, keypoints))
        rotated_keypoints = {}

        # if not valid, skip to the next keypoints
        if not is_valid(keypoints=keypoints):
            continue

        # for one person
        person_index += 1
        output_index.append('{}_{}'.format(fname, person_index))

        # calculate the angle of rotation in rad
        reference_point = keypoints['MidHip'] + [0, -100, 0]
        rad, deg = calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'],point2=reference_point)

        # rotate the joints around keypoints['MidHip']
        for key, value in keypoints.items():
            rotated_value = rotate(value, keypoints['MidHip'], rad)
            rotated_keypoints[key] = rotated_value

        # for one person
        for index, triple in enumerate(joint_triples):

            point1 = rotated_keypoints.get(triple[0])
            center = rotated_keypoints.get(triple[1])
            point2 = rotated_keypoints.get(triple[2])

            col_name = '{}_{}_{}'.format(triple[0], triple[1], triple[2])

            if col_name not in output_dict:
                output_dict[col_name] = []

            if point1[2] != 0 and center[2] != 0 and point2[2] != 0:
                rad, deg = calc_angle(point1=point1, center=center, point2=point2)
                output_dict[col_name].append(rad)
            else:
                output_dict[col_name].append(0.0)

        if show:
            for index, pair in enumerate(joint_pairs):
                if keypoints.get(pair[0])[2] != 0 and keypoints.get(pair[1])[2] != 0:
                    # draw original keypoints
                    point1 = (int(keypoints.get(pair[0])[0]), int(keypoints.get(pair[0])[1]))
                    point2 = (int(keypoints.get(pair[1])[0]), int(keypoints.get(pair[1])[1]))
                    cv2.line(image, point1, point2, color=(255, 0, 255), thickness=5)

                    # draw rotated keypoints
                    rotated_point1 = (rotated_keypoints.get(pair[0])[0], rotated_keypoints.get(pair[0])[1])
                    rotated_point2 = (rotated_keypoints.get(pair[1])[0], rotated_keypoints.get(pair[1])[1])
                    cv2.line(image, rotated_point1, rotated_point2, color=(255, 255, 0), thickness=5)

    if show:
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_dict, output_index


if __name__ == '__main__':

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
        df.to_csv('test/test.csv', index=True)
