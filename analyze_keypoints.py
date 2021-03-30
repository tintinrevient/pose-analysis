import numpy as np
import os
import cv2
from scipy import ndimage


# 1. Body 25 keypoints
keypoint_ids = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background']

keypoints_colors = [
    (255., 0., 85.),
    (255., 0., 0.),
    (255., 85., 0.),
    (255., 170., 0.),
    (255., 255., 0.),
    (170., 255., 0.),
    (85., 255., 0.),
    (0., 255., 0.),
    (255., 0., 0.),
    (0., 255., 85.),
    (0., 255., 170.),
    (0., 255., 255.),
    (0., 170., 255.),
    (0., 85., 255.),
    (0., 0., 255.),
    (170., 0., 255.),
    (255., 0., 170.),
    (255., 0., 255.),
    (85., 0., 255.),
    (0., 0., 255.),
    (0., 0., 255.),
    (0., 0., 255.),
    (0., 255., 255.),
    (0., 255., 255.),
    (0., 255., 255.)]

keypoints_pairs = [
    # ('REar', 'REye'), ('LEar', 'LEye'), ('REye', 'Nose'), ('LEye', 'Nose'),
    ('Nose', 'Neck'), ('Neck', 'MidHip'),
    ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('MidHip', 'RHip'), ('MidHip', 'LHip'),
    ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle')]


def euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def calc_angle(point1, center, point2):

    dx1 = point1[0] - center[0]
    dx2 = point2[0] - center[0]
    dy1 = point1[1] - center[1]
    dy2 = point2[1] - center[1]
    m1 = np.sqrt(dx1**2 + dy1**2);
    m2 = np.sqrt(dx2**2 + dy2**2);

    rad = np.arccos((dx1 * dx2 + dy1 * dy2) / (m1 * m2))
    deg = rad * 180.0 / np.pi;

    print('Rad:', rad)
    print('Deg:', deg)

    return rad, deg


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


def load_keypoints(infile, show):

    data = np.load(infile, allow_pickle='TRUE').item()

    print('dimension:', data['dimension'])
    print('number of people:', data['keypoints'].shape[0])

    fname = infile[infile.rfind('/')+1:infile.rfind('_')]
    image = cv2.imread(os.path.join('test', 'pix', '{}.jpg'.format(fname)))

    # iterate through the keypoints of all the people
    for keypoints in data['keypoints']:

        # process the keypoints of one person
        keypoints = dict(zip(keypoint_ids, keypoints))
        rotated_keypoints = {}

        # if not valid, skip to the next keypoints
        if not is_valid(keypoints=keypoints):
            continue

        reference_point = keypoints['MidHip'] + [0, -100, 0]
        rad, deg = calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'],point2=reference_point)

        # rotate the joints around keypoints['MidHip']
        for key, value in keypoints.items():
            rotated_value = rotate(value, keypoints['MidHip'], rad)
            rotated_keypoints[key] = rotated_value

        for index, pair in enumerate(keypoints_pairs):

            if keypoints.get(pair[0])[2] != 0 and keypoints.get(pair[1])[2] != 0:

                # draw original keypoints
                point1 = (int(keypoints.get(pair[0])[0]), int(keypoints.get(pair[0])[1]))
                point2 = (int(keypoints.get(pair[1])[0]), int(keypoints.get(pair[1])[1]))
                image = cv2.line(image, point1, point2, color=(255, 0, 255), thickness=5)

                # draw rotated keypoints
                rotated_point1 = (rotated_keypoints.get(pair[0])[0], rotated_keypoints.get(pair[0])[1])
                rotated_point2 = (rotated_keypoints.get(pair[1])[0], rotated_keypoints.get(pair[1])[1])
                image = cv2.line(image, rotated_point1, rotated_point2, color=(255, 255, 0), thickness=5)

        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

        fname = 81511
        infile = './test/data/{}_keypoints.npy'.format(fname)

        load_keypoints(infile=infile, show=True)