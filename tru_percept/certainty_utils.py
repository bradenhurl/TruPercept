import numpy as np
import os
import sys
import random
import cv2
import logging

from wavedata.tools.obj_detection import obj_utils
from avod.builders.dataset_builder import DatasetBuilder

import perspective_utils
import config as cfg
import constants as const

class BoxObj():
    def __init__(self, u, v, w, rearBotLeft, frontBotLeft, rearTopLeft, rearBotRight):
        self.u = u
        self.v = v
        self.w = w
        self.rearBotLeft = rearBotLeft
        self.frontBotLeft = frontBotLeft
        self.rearTopLeft = rearTopLeft
        self.rearBotRight = rearBotRight

# Right now this actually loads number of points in 3D box
def load_certainties(persp_dir, idx):
    filepath = persp_dir + '/certainty/{:06d}.txt'.format(idx)

    if os.path.exists(filepath):
        with open(filepath, 'r') as fid:
            data = np.loadtxt(fid)
            data_array = np.array(data, ndmin=1)
            return data_array

    return []

# See certainty eqn in paper
def certainty_from_num_3d_points(num_points):
    return min(1.0, (max(0, num_points - cfg.gamma_lower) / float(cfg.gamma_upper - cfg.gamma_lower)))


def save_num_points_in_3d_boxes(perspect_dir, additional_cls):

    velo_dir = perspect_dir + 'velodyne'
    certainty_dir = perspect_dir + 'certainty'
    calib_dir = perspect_dir + 'calib'

    open_mode = 'w+'
    if additional_cls:
        open_mode = 'a+'
    elif os.path.exists(certainty_dir):
        for the_file in os.listdir(certainty_dir):
            file_path = os.path.join(certainty_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
                logging.exception(e)

    files = os.listdir(velo_dir)
    num_files = min(len(files), cfg.MAX_IDX - cfg.MIN_IDX)
    file_idx = 0

    if not os.path.exists(certainty_dir):
        os.makedirs(certainty_dir)

    #Estimate each idx
    for file in files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])
        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        point_cloud = get_nan_point_cloud(perspect_dir, idx)

        pred_dir = perspect_dir + "predictions"
        objects = obj_utils.read_labels(pred_dir, idx)#, results=True)

        if objects == None:
            continue
        if point_cloud.shape[1] == 0:
            all_points = read_lidar(filepath)
            if point_cloud.shape[1] == 0:
                logging.critical("Point cloud failed to load!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

        certainty_file = certainty_dir + '/{:06d}.txt'.format(idx)
        with open(certainty_file, open_mode) as f:
            for obj in objects:
                num_points = numPointsIn3DBox(obj, point_cloud, perspect_dir, idx)
                f.write('{}\n'.format(num_points))

            #TODO Speed up by passing list
            # num_points_list = numPointsIn3DBox(objects, point_cloud, perspect_dir, idx)

            # for num_points in num_points_list:
            #     f.write('{}\n'.format(num_points))

        sys.stdout.flush()
        sys.stdout.write('\rFinished point count for idx: {} / {}'.format(
            file_idx, num_files))
        file_idx += 1

def get_nan_point_cloud(perspect_dir, idx):
    calib_dir = perspect_dir + '/calib'
    velo_dir = perspect_dir + '/velodyne'
    all_points = obj_utils.get_lidar_point_cloud(idx, calib_dir, velo_dir)

    # Remove nan points
    nan_mask = ~np.any(np.isnan(all_points), axis=1)
    point_cloud = all_points[nan_mask].T
    return point_cloud

# Takes a point or vector in cam coordinates. Returns it in world coordinates (wc)
def point_to_world(point, gta_position):
    x = np.dot(const.X, gta_position.matrix)
    y = np.dot(const.Y, gta_position.matrix)
    z = np.dot(const.Z, gta_position.matrix)

    matrix = np.vstack((x,y,z))

    rel_pos_GTACam = np.array((point[0], point[2], -point[1])).reshape((1,3))
    rel_pos_WC = np.dot(rel_pos_GTACam, matrix)
    position = gta_position.camPos + rel_pos_WC

    return position.reshape((3,))


# Checks if a point is between two planes created by a perpendicular unit vector and two points
def checkDirection(uVec, point, minP, maxP):
    dotPoint = np.dot(point, uVec)
    dotMax = np.dot(maxP, uVec)
    dotMin = np.dot(minP, uVec)

    if ((dotMax <= dotPoint and dotPoint <= dotMin) or
                (dotMax >= dotPoint and dotPoint >= dotMin)):
        return True

    return False

def in3DBox(point, boxObj, gta_position):
    world_point = point_to_world(point, gta_position)

    if not checkDirection(boxObj.u, world_point, boxObj.rearBotLeft, boxObj.frontBotLeft):
        return False
    if not checkDirection(boxObj.v, world_point, boxObj.rearBotLeft, boxObj.rearTopLeft):
        return False
    if not checkDirection(boxObj.w, world_point, boxObj.rearBotLeft, boxObj.rearBotRight):
        return False

    return True


def numPointsIn3DBox(obj, point_cloud, perspect_dir, img_idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = perspective_utils.load_position(pos_dir, img_idx)

    forward = np.array([obj.l, 0, 0])
    right = np.array([0, obj.w, 0])
    up = np.array([0, 0, obj.h])

    camPos = np.array([gta_position.camPos[0], gta_position.camPos[1], gta_position.camPos[2]])
    forward = point_to_world(forward, gta_position) - gta_position.camPos
    right = point_to_world(right, gta_position) - gta_position.camPos
    up = point_to_world(up, gta_position) - gta_position.camPos


    objPosition = np.array([obj.t[0], obj.t[1], obj.t[2]])
    objWorld = point_to_world(objPosition, gta_position)

    rearBotLeft = np.array([objWorld[0] - forward[0] - right[0] - up[0],
                            objWorld[1] - forward[1] - right[1] - up[1],
                            objWorld[2] - forward[2] - right[2] - up[2]])

    frontBotLeft = np.array([objWorld[0] + forward[0] - right[0] - up[0],
                             objWorld[1] + forward[1] - right[1] - up[1],
                             objWorld[2] + forward[2] - right[2] - up[2]])

    rearTopLeft = np.array([objWorld[0] - forward[0] - right[0] + up[0],
                            objWorld[1] - forward[1] - right[1] + up[1],
                            objWorld[2] - forward[2] - right[2] + up[2]])

    rearBotRight = np.array([objWorld[0] - forward[0] + right[0] - up[0],
                             objWorld[1] - forward[1] + right[1] - up[1],
                             objWorld[2] - forward[2] + right[2] - up[2]])


    u = (frontBotLeft - rearBotLeft) / np.linalg.norm(frontBotLeft - rearBotLeft)
    v = (rearTopLeft - rearBotLeft) / np.linalg.norm(rearTopLeft - rearBotLeft)
    w = (rearBotRight - rearBotLeft) / np.linalg.norm(rearBotRight - rearBotLeft)

    boxObj = BoxObj(u,v,w, rearBotLeft, frontBotLeft, rearTopLeft, rearBotRight)

    point_count = 0

    for idx in range(0, point_cloud.shape[0]):
        if in3DBox(point_cloud[idx, :], boxObj, gta_position):
            point_count = point_count + 1

    return point_count


def read_lidar(filepath):
    """Reads in PointCloud from Kitti Dataset.
        Keyword Arguments:
        ------------------
        velo_dir : Str
                    Directory of the velodyne files.
        img_idx : Int
                  Index of the image.
        Returns:
        --------
        x : Numpy Array
                   Contains the x coordinates of the pointcloud.
        y : Numpy Array
                   Contains the y coordinates of the pointcloud.
        z : Numpy Array
                   Contains the z coordinates of the pointcloud.
        i : Numpy Array
                   Contains the intensity values of the pointcloud.
        [] : if file is not found
        """

    if os.path.exists(filepath):
        with open(filepath, 'rb') as fid:
            data_array = np.fromfile(fid, np.single)

        xyzi = data_array.reshape(-1, 4)

        x = xyzi[:, 0]
        y = xyzi[:, 1]
        z = xyzi[:, 2]
        i = xyzi[:, 3]

        return x, y, z, i
    else:
        return []