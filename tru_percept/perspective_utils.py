import os
import numpy as np
import math
import logging

from wavedata.tools.obj_detection import obj_utils

import trust_utils
import config as cfg
import constants as const
import false_detections as false_dets
import correct_synchronization as synchronize

# Notes on terminology:
# ego_id = The entity_id of the main vehicle which is driving from GTA Collection
# view_id = The id of the vehicle whose viewpoint is being evaluated
# view and perspective are used interchangeably
# view is used for most variable names as it is shorter

X = [1., 0., 0.]
Y = [0., 1., 0.]
Z = [0., 0., 1.]

# Global variable for saving false detections so they don't have to be reloaded
FALSE_DETECTIONS = false_dets.load_false_dets(cfg.DATASET_DIR, \
                        cfg.FALSE_DETECTIONS_SUBDIR, cfg.FALSE_DETECTIONS_TYPE)

class GTAPosition:
    """GTA Position Class
    3    pos          Entity position (Bottom center) x,y,z in GTA world coordinates (in meters)

    3    camPos       Camera position x,y,z in GTA world coordinates (in meters)

    3x3  matrix       Matrix of camera unit vectors (right, up, forward)

    3    forward      Camera forward vector, x, y, z unit vector in GTA world coordinates

    3    right        Camera right vector, x, y, z unit vector in GTA world coordinates

    3    up           Camera up vector, x, y, z unit vector in GTA world coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)
    """

    def __init__(self):
        self.pos = [0., 0., 0.]
        self.camPos = [0., 0., 0.]
        self.matrix = [[1., 0., 0.],
                       [0., -1., 0.],
                       [0., 0., 1.]]

        self.forward = [0., 0., 1.]
        self.right = [1., 0., 0.]
        self.up = [0., -1., 0]

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True


def load_position(pos_dir, idx):
    # Extract the list
    if os.stat(pos_dir + "/%06d.txt" % idx).st_size == 0:
        logging.debug("Failed to load position information!")
        return None

    col_count = 3

    p = np.loadtxt(pos_dir + "/%06d.txt" % idx, delimiter=' ',
                    dtype=str,
                    usecols=np.arange(start=0, step=1, stop=col_count))

    pos = GTAPosition()
    pos.pos[0] = float(p[0, 0])
    pos.pos[1] = float(p[0, 1])
    pos.pos[2] = float(p[0, 2])

    pos.camPos[0] = float(p[1, 0])
    pos.camPos[1] = float(p[1, 1])
    pos.camPos[2] = float(p[1, 2])

    pos.forward[0] = float(p[2, 0])
    pos.forward[1] = float(p[2, 1])
    pos.forward[2] = float(p[2, 2])
    
    pos.right[0] = float(p[3, 0])
    pos.right[1] = float(p[3, 1])
    pos.right[2] = float(p[3, 2])
    
    pos.up[0] = float(p[4, 0])
    pos.up[1] = float(p[4, 1])
    pos.up[2] = float(p[4, 2])

    pos.matrix = np.vstack((pos.right, pos.forward, pos.up))
    return pos

def pc_to_world(pc, perspect_dir, idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, idx)

    x = np.dot(X, gta_position.matrix)
    y = np.dot(Y, gta_position.matrix)
    z = np.dot(Z, gta_position.matrix)

    matrix = np.vstack((x,y,z))

    pc_gta = pc[:,[0, 2, 1]]
    pc_gta[:,2] *= -1
    pc_w = np.dot(pc_gta, matrix)

    pos_arr = np.array(gta_position.camPos)
    pos = np.tile(np.array(gta_position.camPos), (pc_w.shape[0], 1))
    pc_w += pos
    return pc_w

def pc_to_perspective(pc, perspect_dir, idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, idx)

    matrix = gta_position.matrix

    # Extend position to a matrix then subtract it from each point
    pos = np.tile(np.array(gta_position.camPos), (pc.shape[0],1))
    pc -= pos

    # Perform coordinate transformation
    pc_orig = np.dot(pc, matrix)
    pc_gta = np.dot(matrix, pc.T).T

    pc_p = pc_gta[:,[0, 2, 1]]
    pc_p[:,1] *= -1

    return pc_p

def pc_persp_transform(pc, from_persp_dir, to_persp_dir, idx):
    pc_w = pc_to_world(pc, from_persp_dir, idx)
    pc_p = pc_to_perspective(pc_w, to_persp_dir, idx)
    return pc_p

#Changes objects
def to_world(objects, perspect_dir, idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, idx)

    x = np.dot(X, gta_position.matrix)
    y = np.dot(Y, gta_position.matrix)
    z = np.dot(Z, gta_position.matrix)

    matrix = np.vstack((x,y,z))

    for obj in objects:
        rel_pos_GTACam = np.array((obj.t[0], obj.t[2], -obj.t[1])).reshape((1,3))
        rel_pos_WC = np.dot(rel_pos_GTACam, matrix)
        position = gta_position.camPos + rel_pos_WC
        obj.t = (position[0,0], position[0,1], position[0,2])

        #Rotation
        forward_x = np.cos(obj.ry)
        forward_y = np.sin(obj.ry)
        world_forward_x = forward_x * x[0] + forward_y * x[1]
        world_forward_y = forward_x * y[0] + forward_y * y[1]
        rot_y = -np.arctan2(world_forward_y, world_forward_x)
        obj.ry = rot_y


#Changes objects
def to_perspective(objects, perspect_dir, idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, idx)

    mat = gta_position.matrix

    for obj in objects:
        rel_pos_WC = np.array(obj.t).reshape((1,3)) - gta_position.camPos
        rel_pos_GTACam = np.dot(gta_position.matrix, rel_pos_WC.T)
        obj.t = (rel_pos_GTACam[0,0], -rel_pos_GTACam[2,0], rel_pos_GTACam[1,0])

        #Rotation
        forward_x = np.cos(obj.ry)
        forward_y = np.sin(obj.ry)
        world_forward_x = forward_x * mat[0][0] + forward_y * mat[0][1]
        world_forward_y = forward_x * mat[1][0] + forward_y * mat[1][1]
        rot_y = -np.arctan2(world_forward_y, world_forward_x)
        obj.ry = rot_y

# Returns the object of the perspective vehicle (persp_dir)
# Returns it in it's own coordinate system
def get_own_vehicle_object(persp_dir, idx, persp_id):
    ego_dir = persp_dir + '/ego_object/'
    ego_detection = obj_utils.read_labels(ego_dir, idx)
    ego_detection[0].score = 1.0
    ego_detection[0].id = persp_id

    if const.ego_id() == persp_id:
        # These weren't set in this version of synthetic data (TODO)
        ego_detection[0].t = (0, ego_detection[0].h, 0)
        ego_detection[0].ry = -math.pi / 2
    else:
        # Need to convert to perspective coordinates if perspective is not ego vehicle
        # All ego_object objects are in ego-vehicle coordinates
        to_world(ego_detection, get_folder(const.ego_id()), idx)
        to_perspective(ego_detection, persp_dir, idx)

    return ego_detection

# to_persp_dir is the directory of the coordinate frame perspective we want the detections in
# det_persp_dir is the perspective we are obtaining the detections from
# det_persp_id is the ID of the perspective detections are received from
def get_detections(to_persp_dir, det_persp_dir, idx, to_persp_id, det_persp_id,
                   results=False, filter_area=False, override_load=False):
    global FALSE_DETECTIONS

    if cfg.FALSE_DETECTIONS_TYPE != 'random_add_remove' or override_load:
        if results:
            label_dir = det_persp_dir + '/' + cfg.PREDICTIONS_SUBDIR + '/'
        else:
            label_dir = det_persp_dir + '/' + cfg.LABEL_DIR + '/'

        label_path = label_dir + '{:06d}.txt'.format(idx)
        if not os.path.isfile(label_path):
            return []

        ego_detection = get_own_vehicle_object(det_persp_dir, idx, det_persp_id)

        # Note: ego_object will be added in createTrustObjects
        # To ensure its trust/certainty value is set to 1
        if cfg.SYNCHRONIZE_DETS and det_persp_dir != to_persp_dir \
                    and results: # For loading ground truth from source
            # Returns -1 if file doesn't exist, we need to convert this to None
            detections = synchronize.get_synchronized_dets(det_persp_dir, to_persp_dir, idx, ego_detection)
            if detections == -1:
                detections = None
        else:
            detections = ego_detection
            p_detections = obj_utils.read_labels(label_dir, idx, results=results)
            if p_detections is not None:
                for obj in p_detections:
                    detections.append(obj)
    else:
        label_dir = det_persp_dir + cfg.FALSE_DETECTIONS_SUBSUBDIR
        label_path = label_dir + '{:06d}.txt'.format(idx)
        if not os.path.isfile(label_path):
            return []
        detections = obj_utils.read_labels(label_dir, idx, results=True)

    logging.debug("det_persp_id: {} det_persp_dir: {}".format(det_persp_id, det_persp_dir))
    if detections is not None:
        obj_dists = []
        for obj in detections:
            obj_pos = np.asanyarray(obj.t)
            obj_dist = np.sqrt(np.dot(obj_pos, obj_pos.T))
            obj_dists.append(obj_dist)

        if det_persp_dir != to_persp_dir:
            to_world(detections, det_persp_dir, idx)
            to_perspective(detections, to_persp_dir, idx)

        trust_objs = trust_utils.createTrustObjects(det_persp_dir, idx, det_persp_id, detections, results, to_persp_dir, obj_dists)

        if cfg.FALSE_DETECTIONS_TYPE != None and \
                        (override_load or cfg.FALSE_DETECTIONS_TYPE != 'random_add_remove'):
            false_det_list = false_dets.get_false_dets(FALSE_DETECTIONS, \
                                 det_persp_id, idx, cfg.FALSE_DETECTIONS_TYPE, \
                                 to_persp_dir, cfg.DATASET_DIR, trust_objs)
            false_trust_objs = trust_utils.createTrustObjects(det_persp_dir, idx, det_persp_id, false_det_list, results, to_persp_dir)
            for trust_obj in false_trust_objs:
                trust_obj.false_det = True
                trust_objs.append(trust_obj)

        if filter_area:
            to_persp_ego_obj = get_own_vehicle_object(to_persp_dir, idx, to_persp_id)
            trust_objs = filter_labels(trust_objs, trust_objs=True, ego_obj=to_persp_ego_obj[0])

        # Easier for visualizations if returning simple objects
        return trust_objs

    return []

# Returns list of predictions for nearby vehicles
# Includes detections from ego vehicle and from the perspective vehicle of persp_id
# The first list of predictions will be for the persp_id detections
def get_all_detections(idx, persp_id, results, filter_area=False):
    all_perspect_detections = []

    # Load predictions from persp_id vehicle
    persp_dir = get_folder(persp_id)
    perspect_detections = get_detections(persp_dir, persp_dir, idx, persp_id, persp_id, results, filter_area)
    if perspect_detections is not None and len(perspect_detections) > 0:
        all_perspect_detections.append(perspect_detections)

    # Load detections from cfg.DATASET_DIR if ego_vehicle is not the persp_id
    if persp_id != const.ego_id():
        perspect_detections = get_detections(persp_dir, cfg.DATASET_DIR, idx, persp_id, const.ego_id(), results, filter_area)
        if perspect_detections is not None and len(perspect_detections) > 0:
            all_perspect_detections.append(perspect_detections)

    # Load detections from remaining perspectives
    for entity_str in const.valid_perspectives():
        other_persp_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)
        if os.path.isdir(other_persp_dir):
            # Skip own detections since they're loaded first
            if int(entity_str) != persp_id:
                perspect_detections = get_detections(persp_dir, other_persp_dir, idx, persp_id, int(entity_str), results, filter_area)
                if perspect_detections is not None and len(perspect_detections) > 0:
                    all_perspect_detections.append(perspect_detections)

    return all_perspect_detections

def get_folder(persp_id):
    if persp_id == const.ego_id():
        return cfg.DATASET_DIR
    else:
        persp_dir = cfg.DATASET_DIR + '/alt_perspective' + '/{:07d}/'.format(persp_id)
        return persp_dir



#####################################################################
# These are used to filter the detections without having to create a kitti_dataset object
# Based off of filter_lables from avod code
def filter_labels(objects, check_distance=True, max_dist=cfg.MAX_LIDAR_DIST, trust_objs=False, ego_obj=None):
    objects = np.asanyarray(objects)
    filter_mask = np.ones(len(objects), dtype=np.bool)

    for obj_idx in range(len(objects)):
        if trust_objs:
            obj = objects[obj_idx].obj
        else:
            obj = objects[obj_idx]

        if not _check_frustum(obj, ego_obj):
            filter_mask[obj_idx] = False
            continue

        if check_distance and not _check_distance(obj, max_dist):
            filter_mask[obj_idx] = False
            continue

        if not _check_height(obj):
            filter_mask[obj_idx] = False
            continue

    return objects[filter_mask]

# Leave 3m around frustum. Vehicles truncated up to 3m past their
# Centre won't be filtered
SAFETY_FACTOR = 2
def _check_distance(obj, max_dist):
    """This filters an object by distance.
    Args:
        obj: An instance of ground-truth Object Label
    Returns: True or False depending on whether the object
        is less than MAX_LIDAR_DIST + 3 metres away.
        3 is used as a safety factor for objects which
        may be partially truncated
    """

    # Checks total distance
    obj_dist = math.sqrt(obj.t[0]**2 + obj.t[1]**2 + obj.t[2]**2)
    if obj_dist > (max_dist + SAFETY_FACTOR):
        return False

    # For scenario to remove self
    # if obj.t[2] < 2:
    #     return False

    return True

def _check_frustum(obj, ego_obj):
    """This filters an object by frustum and height.
    Args:
        obj: An instance of ground-truth Object Label
    Returns: True or False depending on whether the object
        is within the front frustum and height requirements.
    """

    # Checks if in front of vehicle and if in frustum (and not close to ego-vehicle)
    if obj.t[2] < 0 or abs(obj.t[0]) > obj.t[2] + SAFETY_FACTOR:
        if ego_obj is not None:
            # We want to keep objects close to ego-vehicle for matching and eval purposes
            diff = np.asanyarray(ego_obj.t) - np.asanyarray(obj.t)
            obj_dist = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
            if obj_dist > 1.5 or obj.type != 'Car':
                return False
        else:
            return False

    return True

def _check_height(obj):
    # Check if object is more than 3 metres down 
    if obj.t[1] > 3.0:
        return False

    # If object is more than 3 metres up return False
    if obj.t[1] < -3.0:
        return False

    return True
#####################################################################

# Test function to print out pointcloud mins and maxes
def print_pc_bounds(pc):
    print("Mins: ", np.amin(pc, axis=0))
    print("maxes: ", np.amax(pc, axis=0))