import os
import numpy as np
import math

from wavedata.tools.obj_detection import obj_utils

import certainty_utils
import trust_utils
import config as cfg

# Notes on terminology:
# ego_id = The entity_id of the vehicle which is driving
# view_id = The id of the vehicle whose viewpoint is being evaluated
# view and perspective are used interchangeably
# view is used for most variable names as it is shorter

X = [1., 0., 0.]
Y = [0., 1., 0.]
Z = [0., 0., 1.]

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
        print("Failed to load position information!")
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


# to_persp_dir is the directory of the coordinate frame perspective we want the detections in
# det_persp_dir is the perspective we are obtaining the detections from
# det_persp_id is the ID of the perspective detections are received from
def get_detections(to_persp_dir, det_persp_dir, idx, det_persp_id, results=False, filter_area=False, return_avod_objs=False):
    if results:
        label_dir = det_persp_dir + '/predictions/'
    else:
        label_dir = det_persp_dir + '/label_2/'

    label_path = label_dir + '{:06d}.txt'.format(idx)
    if not os.path.isfile(label_path):
        return []

    detections = obj_utils.read_labels(label_dir, idx, results=results)
    print("det_persp_id: ", det_persp_id, " det_persp_dir: ", det_persp_dir)
    if detections is not None:
        to_world(detections, det_persp_dir, idx)
        to_perspective(detections, to_persp_dir, idx)

        #TODO Verify filter is working
        if filter_area:
            detections = filter_labels(detections)

        # Easier for visualizations if returning simple objects
        if return_avod_objs:
            return detections
        else:
            return trust_utils.createTrustObjects(det_persp_dir, idx, det_persp_id, detections, results)

    return []

# Returns list of predictions for nearby vehicles
# Includes detections from ego vehicle and from the perspective vehicle of persp_id
# The first list of predictions will be for the persp_id detections
def get_all_detections(ego_id, idx, persp_id, results, filter_area=False):
    all_perspect_detections = []

    # Load predictions from persp_id vehicle
    #TODO Test if certainty values are corresponding correctly
    persp_dir = get_folder(ego_id, persp_id)
    predictions_dir = persp_dir + 'predictions'
    preds_file = predictions_dir + '{:06d}.txt'.format(idx)
    if os.path.isfile(preds_file):
        persp_preds = obj_utils.read_labels(predictions_dir, idx, results=True)
    else:
        persp_preds = []
    persp_trust_objs = trust_utils.createTrustObjects(persp_dir, idx, persp_id, persp_preds, results)

    # Load detections from cfg.DATASET_DIR if ego_vehicle is not the persp_id
    if persp_id != ego_id:
        perspect_detections = get_detections(persp_dir, cfg.DATASET_DIR, idx, ego_id, results, filter_area)
        if perspect_detections is not None and len(perspect_detections) > 0:
            all_perspect_detections.append(perspect_detections)

    # Load detections from remaining perspectives
    alt_persp_dir = cfg.DATASET_DIR + '/alt_perspective/'
    for entity_str in os.listdir(alt_persp_dir):
        other_persp_dir = os.path.join(alt_persp_dir, entity_str)
        if os.path.isdir(other_persp_dir):
            # Skip own detections since they're loaded first
            if int(entity_str) != persp_id:
                perspect_detections = get_detections(persp_dir, other_persp_dir, idx, int(entity_str), results, filter_area)
                if perspect_detections is not None and len(perspect_detections) > 0:
                    all_perspect_detections.append(perspect_detections)


    # todo Should test visualizations to ensure all detections
    # are being loaded properly
    # todo - Also change tru_percept to correspond correctly
    return all_perspect_detections

def get_folder(ego_id, persp_id):
    if persp_id == ego_id:
        return cfg.DATASET_DIR
    else:
        persp_dir = cfg.DATASET_DIR + '/alt_perspective' + '/{:07d}/'.format(persp_id)
        return persp_dir



#####################################################################
# These are used to filter the detections without having to create a kitti_dataset object
# Based off of filter_lables from avod code
def filter_labels(objects):
    objects = np.asanyarray(objects)
    filter_mask = np.ones(len(objects), dtype=np.bool)

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]

        if not _check_distance(obj):
            filter_mask[obj_idx] = False
            continue

    return objects[filter_mask]

# Leave 3m around frustrum. Vehicles truncated up to 3m past their
# Centre won't be filtered
SAFETY_FACTOR = 3
MAX_LIDAR_DIST = 80
def _check_distance(obj):
    """This filters an object by distance and frustrum.
    Args:
        obj: An instance of ground-truth Object Label
    Returns: True or False depending on whether the object
        is less than MAX_LIDAR_DIST + 3 metres away.
        3 is used as a safety factor for objects which
        may be partially truncated
    """

    print("Object position: ", obj.t)
    # Checks if in front of vehicle
    if obj.t[2] < 0:
        return False

    # Checks if in frustrum
    if abs(obj.t[0]) > obj.t[2] + SAFETY_FACTOR:
        return False

    # Checks total distance
    obj_dist = math.sqrt(obj.t[0]**2 + obj.t[1]**2 + obj.t[2]**2)
    if obj_dist > (MAX_LIDAR_DIST + SAFETY_FACTOR):
        return False

    print("??????????????????????????????????????Not filtered")
    return True
#####################################################################