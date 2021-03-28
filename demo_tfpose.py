import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import os
from pathlib import Path


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0, 0)
    return (0, 0)


def euclidian(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_calc(p0, p1, p2):
    '''
        p1 is center point from where we measured angle between p0 and
    '''
    try:
        a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
        b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
        angle = math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi
    except:
        return 0
    return int(angle)


def plank(a, b, c, d, e, f):
    # There are ranges of angle and distance to for plank.
    '''
        a and b are angles of hands
        c and d are angle of legs
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if (a in range(50, 100) or b in range(50, 100)) and (c in range(135, 175) or d in range(135, 175)) and (
            e in range(50, 250) or f in range(50, 250)):
        return True
    return False


def mountain_pose(a, b, c, d, e):
    '''
        a is distance between two wrists
        b and c are angle between neck,shoulder and wrist
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if a in range(20, 160) and b in range(60, 140) and c in range(60, 140) and d in range(100, 145) and e in range(100,
                                                                                                                   145):
        return True
    return False


def draw_str(dst, xxx_todo_changeme, s, color, scale):
    (x, y) = xxx_todo_changeme
    if (color[0] + color[1] + color[2] == 255 * 3):
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness=4, lineType=10)
    else:
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness=4, lineType=10)
    # cv2.line
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)


def estimate_pose(infile, show):

    print('input:', infile)

    # cap = cv2.VideoCapture(args.input)
    # hasImage, image = cap.read()
    image = cv2.imread(infile)

    # generate out directory and out file
    outfile = generate_outfile(infile=infile)

    estimator = TfPoseEstimator(get_graph_path('cmu'), target_size=(width, height))

    humans = estimator.inference(image, resize_to_default=True, upsample_size=4.0)
    pose = humans
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    num_of_humans = len(humans)
    print("Total number of humans : ", num_of_humans)

    if show:
        cv2.imshow('tf-pose-estimation result', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(outfile, image)
        print('output:', outfile)


def generate_outfile(infile):

    outdir = os.path.join('output', infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:]
    outfile = os.path.join('output', fname)

    return outfile


if __name__ == '__main__':

    orange_color = (0, 140, 255)

    parser = argparse.ArgumentParser(description='tf-pose')
    parser.add_argument('--input', help='Path to image or video')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % ('cmu', get_graph_path('cmu')))
    width, height = model_wh('432x368')

    if os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            estimate_pose(infile=str(path), show=False)

    elif os.path.isfile(args.input):
        estimate_pose(infile=args.input, show=False)
    else:
        pass

    # # distance calculations for mountain_pose
    # head_hand_dst_l = int(euclidian(find_point(pose, 0), find_point(pose, 7)))
    # head_hand_dst_r = int(euclidian(find_point(pose, 0), find_point(pose, 4)))
    # m_pose = int(euclidian(find_point(pose, 7), find_point(pose, 4)))
    # # angle calcucations
    # angle1 = angle_calc(find_point(pose, 6), find_point(pose, 5), find_point(pose, 1))
    # angle5 = angle_calc(find_point(pose, 3), find_point(pose, 2), find_point(pose, 1))
    #
    # if mountain_pose(m_pose, angle1, angle5, head_hand_dst_r, head_hand_dst_l):
    #     action = "Mountain Pose"
    #     is_yoga = True
    #     draw_str(image, (20, 50), action, orange_color, 2)
    #     logger.debug("*** Mountain Pose ***")

    # # distance calculations for plank
    # head_hand_dst_l = int(euclidian(find_point(pose, 0), find_point(pose, 7)))
    # head_hand_dst_r = int(euclidian(find_point(pose, 0), find_point(pose, 4)))
    # # angle calcucations
    # angle2 = angle_calc(find_point(pose, 7), find_point(pose, 6), find_point(pose, 5))
    # angle4 = angle_calc(find_point(pose, 11), find_point(pose, 12), find_point(pose, 13))
    # angle6 = angle_calc(find_point(pose, 4), find_point(pose, 3), find_point(pose, 2))
    # angle8 = angle_calc(find_point(pose, 8), find_point(pose, 9), find_point(pose, 10))
    #
    # if plank(angle2, angle6, angle4, angle8, head_hand_dst_r, head_hand_dst_l):
    #     action = "Plank"
    #     is_yoga = True
    #     draw_str(image, (20, 50), " Plank", orange_color, 2)
    #     logger.debug("*** Plank ***")

    # body ratio
    # Total_body_r = int(euclidian(find_point(pose, 0), find_point(pose, 10)))  # Right height
    # Total_body_l = int(euclidian(find_point(pose, 0), find_point(pose, 13)))  # Left height
    # Leg_r = int(euclidian(find_point(pose, 8), find_point(pose, 10)))  # Right leg
    # Leg_l = int(euclidian(find_point(pose, 11), find_point(pose, 13)))  # Left leg
    #
    # print(Leg_l, Total_body_l)
    # print(Leg_r, Total_body_r)
    #
    # try:
    #     LBR_l = round(Leg_l / Total_body_l, 2)
    #     LBR_r = round(Leg_r / Total_body_r, 2)
    #     average_ratio = round((LBR_l + LBR_r) / 2, 3)
    # except:
    #     pass
    #
    # draw_str(image, (20, 80), "leg to body ratio = " + str(average_ratio), orange_color, 1.5)