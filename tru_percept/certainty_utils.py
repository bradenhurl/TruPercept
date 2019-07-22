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
def certainty_from_3d_points(num_points, obj_type):
    gamma_l = cfg.gamma_lower
    gamma_u = cfg.gamma_upper
    if obj_type == 'Pedestrian':
        gamma_l = cfg.gamma_lower_peds
        gamma_u = cfg.gamma_upper_peds

    return min(1.0, (max(0, num_points - gamma_l) / float(gamma_u - gamma_l)))