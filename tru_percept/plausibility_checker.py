import numpy as np

from wavedata.tools.obj_detection import obj_utils

import config as cfg
import perspective_utils as p_utils
import points_in_3d_boxes as points_3d
from tools.visualization import vis_utils

# Check if obj is plausible from ego-vehicle perspective
def is_plausible(obj, idx, detector_id, det_idx):

    # Can only deny plausibility in the fov of the ego-vehicle sensors
    # Filter objects to be in the sensor range, return True if out of range/fov
    filtered_obj = p_utils.filter_labels([obj])
    if len(filtered_obj) == 0:
        return True

    # Check if object is more than MAX_LIDAR_DIST away
    # If yes then it is plausible since our sensors don't go that far
    obj_pos = np.asanyarray(obj.t)
    obj_dist = np.dot(obj_pos, obj_pos.T)
    if obj_dist >= cfg.MAX_LIDAR_DIST:
        return True

    # TODO - Do we want to check this or is it better to only check frustum?
    # This may allow untrustworthy entities to put ground points in box
    # Check if points in 3D box first, if yes it is plausible
    # points_dict = points_3d.load_points_in_3d_boxes(idx, const.ego_id())
    # points_in_box = points_dict[detector_id, det_idx]
    # if points_in_box > 0:
    #     return True

    # Load point cloud
    velo_dir = cfg.DATASET_DIR + '/velodyne'
    calib_dir = cfg.DATASET_DIR + '/calib'
    pc = obj_utils.get_lidar_point_cloud(idx, calib_dir, velo_dir)

    # Obtain unit vector to object
    unit_vec_forward = obj_pos / np.linalg.norm(obj_pos)

    # The size of the frustum square at the object's distance
    # We want to ensure we are within the object's bounding box
    half_size = min(obj.l / 4.0, obj.w / 4.0)

    # camera position is origin
    cam_pos = np.array([0,0,0])

    # Calculate the up and right unit vectors for the frustum
    y_up = (obj_pos[0] ** 2 + obj_pos[2] ** 2) / obj_pos[1]
    vec_down = np.array([obj_pos[0], y_up, obj_pos[2]])
    unit_vec_down = vec_down / np.linalg.norm(vec_down)
    unit_vec_right = np.cross(unit_vec_down, unit_vec_forward)

    # KITTI labels are at bottom center of 3D bounding box
    obj_center = obj_pos - np.array([0, obj.h / 2, 0])

    # Obtain 2D box for computing frustum boundary planes
    top_left = -(half_size * unit_vec_down) - (half_size * unit_vec_right) + obj_center
    top_right = -(half_size * unit_vec_down) + (half_size * unit_vec_right) + obj_center
    bot_left = (half_size * unit_vec_down) - (half_size * unit_vec_right) + obj_center
    bot_right = (half_size * unit_vec_down) + (half_size * unit_vec_right) + obj_center

    # Compute the 4 boundary planes
    plane_top = get_plane_params(cam_pos, top_left, top_right)
    plane_bot = get_plane_params(cam_pos, bot_left, bot_right)
    plane_right = get_plane_params(cam_pos, bot_right, top_right)
    plane_left = get_plane_params(cam_pos, bot_left, top_left)

    # Extend pc with ones for dot product with planes
    shape = pc.shape
    pc = pc.T
    new_pc = np.ones((pc.shape[0], pc.shape[1] + 1))
    new_pc[:,:-1] = pc
    pc = new_pc

    # Filter the points for each plane
    pc = pc[np.dot(pc, plane_top) < 0]
    pc = pc[np.dot(pc, plane_bot) > 0]
    pc = pc[np.dot(pc, plane_right) > 0]
    pc = pc[np.dot(pc, plane_left) < 0]

    # If there are no points left then the object is not plausible
    # Otherwise LiDAR should've hit something on or before the object
    if pc.shape[0] == 0:
        return False

    # Remove extended column
    pc = pc[:, 0:3]

    # To visualize the resulting points
    # vis_utils.vis_pc(pc.T, [obj])

    # Save the current number of points before distance based culling
    num_points = pc.shape[0]

    # Compute if remaining points are closer or further than the object centre
    point_distances = np.sum(np.multiply(pc, pc), axis=1)
    pc_closer = pc[point_distances < obj_dist]
    num_closer_points = pc_closer.shape[0]

    # To visualize the resulting points
    # vis_utils.vis_pc(pc_closer.T, [obj])

    # Return True if > 10% of points are closer
    if num_closer_points / float(num_points) > 0.1:
        return True

    return False

# Returns the parameters of a plane from 3 points
# Equation of plane is ax + by + cz = d
def get_plane_params(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return a, b, c, d