import os
import sys
import random
import logging
import cv2
import numpy as np

from wavedata.tools.obj_detection import obj_utils
from avod.builders.dataset_builder import DatasetBuilder

import config as cfg

testing = False
#Which method to use for determining plane coefficients
# 0 - Ground truth points
# 1 - RANSAC
# 2 - Jason's Autonomoose method
use_ground_points = 0
ransac = 1
moose = 2

def estimate_ground_planes(base_dir, dataset_config, plane_method=0, specific_idx=-1):
    velo_dir = base_dir + 'velodyne'
    plane_dir = base_dir + 'planes'
    if plane_method == 1:
        plane_dir = plane_dir + '_ransac'
    ground_points_dir = base_dir + 'ground_points'
    grid_points_dir = base_dir + 'ground_points_grid'
    calib_dir = base_dir + 'calib'

    files = os.listdir(velo_dir)
    num_files = len(files)
    file_idx = 0

    #For checking ground planes
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                     use_defaults=False)
    kitti_utils = dataset.kitti_utils

    if not os.path.exists(plane_dir):
        os.makedirs(plane_dir)

    #Estimate each idx
    for file in files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        if specific_idx != -1:
            idx = specific_idx
        planes_file = plane_dir + '/%06d.txt' % idx
        logging.debug("Index: %d", idx)

        lidar_point_cloud = obj_utils.get_lidar_point_cloud(idx, calib_dir, velo_dir)
        # Reshape points into N x [x, y, z]
        point_cloud = np.array(lidar_point_cloud).transpose().reshape((-1,3)).T

        ground_points_failed = False
        if plane_method == use_ground_points:
            s = loadGroundPointsFromFile(idx, ground_points_dir, grid_points_dir)
            if s.shape[0] < 4:
                logging.debug("Not enough points at idx: %d", idx)
                ground_points_failed = True
            else:
                m = estimate(s)
                a, b, c, d = m
                plane = loadKittiPlane(m)
                #ground_points_failed = checkBadSlices(point_cloud, plane, kitti_utils)
        
        if plane_method == ransac or ground_points_failed:
            logging.debug("PC shape: {}".format(point_cloud.shape))
            points = point_cloud.T
            all_points_near = points[
                (points[:, 0] > -3.0) &
                (points[:, 0] < 3.0) &
                (points[:, 1] > -3.0) &
                (points[:, 1] < 0.0) &
                (points[:, 2] < 20.0) &
                (points[:, 2] > 2.0)]
            n = all_points_near.shape[0]
            logging.debug("Number of points near: %d", n)
            max_iterations = 100
            goal_inliers = n * 0.5
            m, b = run_ransac(all_points_near, lambda x, y: is_inlier(x, y, 0.2), 3, goal_inliers, max_iterations)
            a, b, c, d = m
        elif plane_method == moose:
            plane_coeffs = estimate_plane_coeffs(point_cloud.T)

        with open(planes_file, 'w+') as f:
            f.write('# Plane\nWidth 4\nHeight 1\n')
            if plane_method == ransac or plane_method == use_ground_points:
                coeff_string = '%.6e %.6e %.6e %.6e' % (a,b,c,d)
            else:
                coeff_string = '%.6e %.6e %.6e %.6e' % (plane_coeffs[0], plane_coeffs[1], plane_coeffs[2], plane_coeffs[3])
            f.write(coeff_string)

        sys.stdout.write("\rGenerating plane {} / {}".format(
            file_idx + 1, num_files))
        sys.stdout.flush()
        file_idx = file_idx + 1

        if testing and idx == 2 or specific_idx != -1:
            quit()

#Modified from obj_utils.py in wavedata
def loadKittiPlane(plane_coeffs):
    plane = np.asarray(plane_coeffs)

    # Ensure normal is always facing up.
    # In Kitti's frame of reference, +y is down
    if plane[1] > 0:
        plane = -plane

    # Normalize the plane coefficients
    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm
    return plane

#Returns true (bad) if any slices doesn't hit at least one point (otherwise it will fail in AVOD)
def checkBadSlices(point_cloud, ground_plane, kitti_utils):
    num_slices = kitti_utils.bev_generator.num_slices
    height_hi_set = kitti_utils.bev_generator.height_hi
    height_lo_set = kitti_utils.bev_generator.height_lo
    height_per_division = (height_hi_set - height_lo_set) / num_slices
    for slice_idx in range(num_slices):

        height_lo = height_lo_set + slice_idx * height_per_division
        height_hi = height_lo + height_per_division

        slice_filter = kitti_utils.create_slice_filter(
            point_cloud,
            kitti_utils.area_extents,
            ground_plane,
            height_lo,
            height_hi)

        # Apply slice filter
        slice_points = point_cloud.T[slice_filter]

        if len(slice_points) <= 1:
            logging.debug("Found bad slice!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return True

    return False

def loadGroundPointsFromFile(idx, ground_points_dir, grid_points_dir):
    file = ground_points_dir + '/%06d.txt' % idx
    p = np.loadtxt(file, delimiter=',',
                       dtype=float,
                       usecols=np.arange(start=0, step=1, stop=3))

    x = p[:,0]
    y = p[:,1]
    z = p[:,2]

    valid_mask = z > -5
    use_extra_points = np.sum(valid_mask) <= 4
    if use_extra_points:
        extra_points = loadPointsFromGrid(idx, grid_points_dir, valid_mask, x, y)

    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    all_points = np.vstack((y, -z, x)).T

    if use_extra_points:
        all_points = np.vstack((all_points,extra_points))
    return all_points

def loadPointsFromGrid(idx, grid_points_dir, valid_mask, x, y):
    file = grid_points_dir + '/%06d.txt' % idx
    p = np.loadtxt(file, delimiter=',',
                       dtype=float,
                       usecols=np.arange(start=0, step=1, stop=3))

    #Shrink to only contain points within smaller grid in front of vehicle
    x = p[:,0]
    y = p[:,1]
    z = p[:,2]

    #x is forward here, y is right, z is up
    mask = (x > 0) & (x < 40)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    mask = (y > -15) & (y < 15)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    mask = z > -5
    x = x[mask]
    y = y[mask]
    z = z[mask]

    points = np.vstack((y, -z, x)).T
    np.set_printoptions(suppress=True)

    return points


def getGridIndex(x,y):
    max_dist = 120
    interval = 2
    row = (2*max_dist)/2 + 1
    idx = max_dist + x/interval + (y/interval + 1)*row
    return idx

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

def estimate_plane_coeffs(points):
    """Calculates least squares fit of a plane on a set of points and returns
    plane coefficients
    Args:
        points: points (N, 3)
    Returns:
        plane coefficients
    """
    all_points = np.vstack((points, [0.0, 1.653, 0.0]))
    centroid = np.mean(all_points, axis=0)
    shifted = points - centroid

    points_x = shifted[:, 0]
    points_y = shifted[:, 1]
    points_z = shifted[:, 2]

    # Sums
    xx = np.dot(points_x, points_x)
    xy = np.dot(points_x, points_y)
    xz = np.dot(points_x, points_z)
    yy = np.dot(points_y, points_y)
    yz = np.dot(points_y, points_z)
    zz = np.dot(points_z, points_z)

    det_x = yy * zz - yz * yz
    det_y = xx * zz - xz * xz
    det_z = xx * yy - xy * xy

    det_max = max(det_x, det_y, det_z)

    if det_max == det_x:
        normal = [det_x, xz * yz - xy * zz, xy * yz - xz * yy]
    elif det_max == det_y:
        normal = [xz * yz - xy * zz, det_y, xy * xz - yz * xx]
    else:
        normal = [xy * yz - xz * yy, xy * xz - yz * xx, det_z]

    normal = -(normal / np.linalg.norm(normal))
    d = -(normal[0] * centroid[0] +
          normal[1] * centroid[1] +
          normal[2] * centroid[2])
    return np.hstack([normal, d])


def estimate_ground_plane(point_cloud):
    """Estimates a ground plane by subsampling 2048 points in an area in front
    of the car, and running a least squares fit of a plane on the lowest
    points along y.
    Args:
        point_cloud: point cloud (3, N)
    Returns:
        ground_plane: ground plane coefficients
    """

    if len(point_cloud) == 0:
        raise ValueError('Lidar points are completely empty')

    # Subsamples points in from of the car, 10m across and 30m in depth
    points = point_cloud.T
    all_points_near = points[
        (points[:, 0] > -5.0) &
        (points[:, 0] < 5.0) &
        (points[:, 2] < 30.0) &
        (points[:, 2] > 2.0)]

    if len(all_points_near) == 0:
        raise ValueError('No Lidar points in this frame')

    # Subsample near points
    subsample_num_near = 2048
    near_indices = np.random.randint(0, len(all_points_near),
                                     subsample_num_near)
    points_subsampled = all_points_near[near_indices]

    # Split into distance bins
    all_points_in_bins = []
    all_cropped_points = []
    for dist_bin_idx in range(3):
        points_in_bin = points_subsampled[
            (points_subsampled[:, 2] > dist_bin_idx * 10.0) &
            (points_subsampled[:, 2] < (dist_bin_idx + 1) * 10.0)]

        # Save to points in bins
        all_points_in_bins.extend(points_in_bin)

        # Sort by y for cropping
        # sort_start_time = time.time()
        y_order = np.argsort(points_in_bin[:, 1])

        # Crop each bin
        num_points_in_bin = len(points_in_bin)
        # crop_start_time = time.time()

        crop_indices = np.array([int(num_points_in_bin * 0.90),
                                 int(num_points_in_bin * 0.98)])
        points_cropped = points_in_bin[
            y_order[crop_indices[0]:crop_indices[1]]]

        all_cropped_points.extend(points_cropped)
    all_cropped_points = np.asarray(all_cropped_points)

    # Do least squares fit to get ground plane coefficients
    ground_plane = estimate_plane_coeffs(all_cropped_points)

    return ground_plane

########################
#RANSAC code from: https://github.com/falcondai/py-ransac
########################
def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def run_ransac(data, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    logging.debug('took iterations: {}, best model: {}, explains: {}'.format(i+1,best_model,best_ic))
    return best_model, best_ic