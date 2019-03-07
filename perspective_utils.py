import numpy as np
import math
import os
from wavedata.tools.obj_detection import obj_utils

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


def load_position(pos_dir, img_idx):
    # Extract the list
    if os.stat(pos_dir + "/%06d.txt" % img_idx).st_size == 0:
        print("Failed to load position information!")
        return None

    col_count = 3

    p = np.loadtxt(pos_dir + "/%06d.txt" % img_idx, delimiter=' ',
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
def to_world(objects, perspect_dir, img_idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, img_idx)

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
def to_perspective(objects, perspect_dir, img_idx):
    pos_dir = perspect_dir + '/position_world/'
    gta_position = load_position(pos_dir, img_idx)

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


def get_detections(main_perspect_dir, altPerspect_dir, img_idx, entity_str, results=False):
    perspect_idx = int(entity_str)
    perspect_dir = altPerspect_dir + entity_str
    if results:
        label_dir = perspect_dir + '/predictions/'
    else:
        label_dir = perspect_dir + '/label_2/'

    label_path = label_dir + '{:06d}.txt'.format(img_idx)
    if not os.path.isfile(label_path):
        return None

    detections = obj_utils.read_labels(label_dir, img_idx, results=results)

    if detections != None:
        to_world(detections, perspect_dir, img_idx)
        to_perspective(detections, main_perspect_dir, img_idx)

    return detections

# Retrieves list of predictions for nearby vehicles if they 
def get_all_detections(main_perspect_dir, idx, results):
    all_perspect_detections = []

    altPerspect_dir = main_perspect_dir + '/alt_perspective/'

    for entity_str in os.listdir(altPerspect_dir):
        if not os.path.isdir(os.path.join(altPerspect_dir, entity_str)):
            continue
        
        perspect_detections = get_detections(main_perspect_dir, altPerspect_dir, idx, entity_str, results)
        all_perspect_detections.append(perspect_detections)

    return all_perspect_detections