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

def load_certainties(c_dir, idx):
    filepath = c_dir + '/certainty/{:06d}.txt'.format(idx)

    if os.path.exists(filepath):
        with open(filepath, 'r') as fid:
            data_array = np.loadtxt(fid)
            data_array.reshape(-1)
            return data_array

    return []

def save_num_points_in_3d_boxes(base_dir, additional_cls):

    velo_dir = base_dir + 'velodyne'
    certainty_dir = base_dir + 'certainty'
    calib_dir = base_dir + 'calib'

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
        objects = obj_utils.read_labels(pred_dir, idx)#, results=True)

        if objects == None:
            continue
        if point_cloud.shape[1] == 0:
            print("Base dir: ", base_dir)
            print("Point cloud failed to load!!!!!!!!!!!!!!!!!!!!!!!!")
            all_points = read_lidar(filepath)
            if point_cloud.shape[1] == 0:
                print("Point cloud failed to load2!!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            print("Point cloud load2 successful!!!!!!!!!!!!!!!!!!!!!!!!")

        certainty_file = certainty_dir + '/{:06d}.txt'.format(idx)
        with open(certainty_file, open_mode) as f:
            for obj in objects:
                num_points = numPointsIn3DBox(obj, point_cloud, base_dir, idx)
                f.write('{}\n'.format(num_points))

            #TODO Speed up by passing list
            # num_points_list = numPointsIn3DBox(objects, point_cloud, base_dir, idx)

            # for num_points in num_points_list:
            #     f.write('{}\n'.format(num_points))
        

# Takes a point or vector in cam coordinates. Returns it in world coordinates (wc)
def point_to_world(point, gta_position):
    x = np.dot(X, gta_position.matrix)
    y = np.dot(Y, gta_position.matrix)
    z = np.dot(Z, gta_position.matrix)

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

    # print("dotmin, point, max: ", dotMin, dotPoint, dotMax)

    if ((dotMax <= dotPoint and dotPoint <= dotMin) or
                (dotMax >= dotPoint and dotPoint >= dotMin)):
        return True

    return False

def in3DBox(point, obj, gta_position):
    world_point = point_to_world(point, gta_position)

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

    # print("Forward: ", forward)
    # print("Obj world: ", objWorld)
    # print("world point: ", world_point)
    # print("Obj: ", objWorld)
    # print("RearBotLeft: ", rearBotLeft)
    # print("u,v,w: ", u, v, w)

    if not checkDirection(u, world_point, rearBotLeft, frontBotLeft):
        return False
    if not checkDirection(v, world_point, rearBotLeft, rearTopLeft):
        return False
    if not checkDirection(w, world_point, rearBotLeft, rearBotRight):
        return False

    return True


def numPointsIn3DBox(obj, point_cloud, perspect_dir, img_idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = perspective_utils.load_position(pos_dir, img_idx)

    point_count = 0

    for idx in range(0, point_cloud.shape[0]):
        if in3DBox(point_cloud[idx, :], obj, gta_position):
            point_count = point_count + 1

    # For testing individual points
    # point = np.array([obj.t[0], obj.t[1]+ (obj.l - 1)/2, obj.t[2]])
    # result = in3DBox(point, obj, gta_position)
    # print("Result is: ", result)

    print("Point count: ", point_count)

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