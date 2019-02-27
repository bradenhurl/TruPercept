import numpy as np
import os
import sys
import random
import cv2
from wavedata.tools.obj_detection import obj_utils
from avod.builders.dataset_builder import DatasetBuilder
import perspective_utils

X = [1., 0., 0.]
Y = [0., 1., 0.]
Z = [0., 0., 1.]

def save_num_points_in_3d_boxes(base_dir):
    velo_dir = base_dir + 'velodyne'
    certainty_dir = base_dir + 'certainty'
    labels_dir = base_dir + 'predictions'
    calib_dir = base_dir + 'calib'

    files = os.listdir(velo_dir)
    num_files = len(files)
    file_idx = 0

    if not os.path.exists(certainty_dir):
        os.makedirs(certainty_dir)

    #Estimate each idx
    for file in files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        all_points = obj_utils.get_lidar_point_cloud(idx, calib_dir, velo_dir)

        # Remove nan points
        nan_mask = ~np.any(np.isnan(all_points), axis=1)
        point_cloud = all_points[nan_mask].T
        print("PC shape: ", point_cloud.shape)

        pred_dir = base_dir + "predictions"
        objects = obj_utils.read_labels(pred_dir, idx, results=True)

        if objects == None:
            continue

        certainty_file = certainty_dir + '/{:06d}.txt'.format(idx)
        with open(certainty_file, 'w+') as f:
            for obj in objects:
                num_points = numPointsIn3DBox(obj, point_cloud, base_dir, idx)

                f.write('{}\n'.format(num_points))
        

# Takes a point or vector in cam coordinates. Returns it in world coordinates (wc)
def point_to_world(point, gta_position):
    x = np.dot(X, gta_position.matrix)
    y = np.dot(Y, gta_position.matrix)
    z = np.dot(Z, gta_position.matrix)

    matrix = np.vstack((x,y,z))

    # Velodyne is x forward, y left, z up
    #TODO ensure points are coming from velodyne and this is correct
    rel_pos_GTACam = np.array((point[0], point[2], -point[1])).reshape((1,3))
    rel_pos_WC = np.dot(rel_pos_GTACam, matrix)
    position = gta_position.camPos + rel_pos_WC
    point_wc = (position[0,0], position[0,1], position[0,2])

    return point_wc


# Checks if a point is between two planes created by a perpendicular unit vector and two points
def checkDirection(uVec, point, minP, maxP):
    dotPoint = np.dot(point, uVec)
    dotMax = np.dot(maxP, uVec)
    dotMin = np.dot(minP, uVec)

    if ((dotMax <= dotPoint and dotPoint <= dotMin) or
                (dotMax >= dotPoint and dotPoint >= dotMin)):
        return True

    return False

def in3DBox(point, obj, gta_position):
    world_point = point_to_world(point, gta_position)

    forward = np.array([obj.l, 0, 0])
    right = np.array([0, obj.w, 0])
    up = np.array([0, 0, obj.h])

    forward = point_to_world(forward, gta_position)
    right = point_to_world(right, gta_position)
    up = point_to_world(up, gta_position)

    rearBotLeft = np.array([obj.t[0] - forward[0] - right[0] - up[0],
                            obj.t[1] - forward[1] - right[1] - up[1],
                            obj.t[2] - forward[2] - right[2] - up[2]])

    print(rearBotLeft.shape)


def numPointsIn3DBox(obj, point_cloud, perspect_dir, img_idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = perspective_utils.load_position(pos_dir, img_idx)

    point_count = 0

    for idx in range(0, point_cloud.shape[0]):
        if in3DBox(point_cloud[idx, :], obj, gta_position):
            point_count = point_count + 1

    return point_count