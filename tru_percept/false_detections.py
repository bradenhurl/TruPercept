import numpy as np
import os
import math
import copy
import random

from wavedata.tools.obj_detection import obj_utils

import config as cfg
import std_utils
import perspective_utils as p_utils

# Notes:
# 1. Objects should be entered from the ego-vehicle perspective

# Returns a dictionary with the false detections
# Call dictionary with vehicle id
def load_false_dets(dataset_dir, false_dets_subdir, false_dets_method_str):
    # Define the dictionary
    det_dict = {}

    if false_dets_method_str == None:
        return {}

    # Random vehicle choices should be kept consistent between random methods
    if false_dets_method_str == 'random_add_remove' or \
            false_dets_method_str == 'random_add' or \
            false_dets_method_str == 'random_remove':
        false_dets_method_str = 'random_{}'.format(cfg.RANDOM_MALICIOUS_PROBABILITY)

    filepath = dataset_dir + '/' + false_dets_subdir + '/' + \
               false_dets_method_str + '.txt'

    if not os.path.isfile(filepath):
        print("Could not find false detections filepath: ", filepath)
        return {}

    # Extract the list
    if os.stat(filepath).st_size == 0:
        print("Filesize 0 for false detections filepath: ", filepath)
        return {}

    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            det_dict[int(line)] = True

    # To use in case we want to add more complicated false detection files
    # p = np.loadtxt(filepath, delimiter=' ',
    #                dtype=str,
    #                usecols=np.arange(start=0, step=1, stop=1))

    # # Check if the output is single dimensional or multi dimensional
    # if len(p.shape) > 1:
    #     label_num = p.shape[0]
    # else:
    #     label_num = 1

    # for idx in np.arange(label_num):

    #     if label_num > 1:
    #         det_dict[int(p[idx,0])] = True
    #     else:
    #         det_dict[int(p[0])] = True

    return det_dict

def get_false_dets(false_dets_dict, persp_id, idx, false_dets_method_str,
                    to_persp_dir, dataset_dir, trust_objs):

    if int(persp_id) not in false_dets_dict:
        return []

    # For testing vehicle trust, allows vehicle to accrue positive trust score
    if idx < cfg.FALSE_DETS_START_IDX:
        return []

    # Return a vehicle 3 metres in front of the ego vehicle
    if false_dets_method_str == 'malicious_front':
        ego_dir = dataset_dir + '/ego_object'
        ego_detection = obj_utils.read_labels(ego_dir, idx)
        ego_detection[0].score = 1.0
        # These weren't set in this version of synthetic data (TODO)
        ego_detection[0].t = (0, ego_detection[0].h, 8)
        ego_detection[0].ry = math.pi / 2

        # Use dataset_dir to put the object in front of the ego-vehicle
        p_utils.to_world(ego_detection, dataset_dir, idx)
        p_utils.to_perspective(ego_detection, to_persp_dir, idx)
        return ego_detection
    # Return a vehicle 3 metres in front of the ego vehicle
    elif false_dets_method_str == 'many_malicious_front':
        ego_dir = dataset_dir + '/ego_object'
        ego_detection = obj_utils.read_labels(ego_dir, idx)
        ego_detection[0].score = 1.0
        # These weren't set in this version of synthetic data (TODO)
        ego_detection[0].t = (0, ego_detection[0].h, 0)
        ego_detection[0].ry = math.pi / 2

        new_obj = copy.deepcopy(ego_detection[0])
        new_obj.t = (-4, ego_detection[0].h, 8)
        ego_detection.append(new_obj)

        new_obj_2 = copy.deepcopy(ego_detection[0])
        new_obj_2.t = (-2, ego_detection[0].h, 15)
        ego_detection.append(new_obj_2)

        # Set the position of the false detection after copying
        ego_detection[0].t = (0, ego_detection[0].h, 8)

        # Use dataset_dir to put the object in front of the ego-vehicle
        p_utils.to_world(ego_detection, dataset_dir, idx)
        p_utils.to_perspective(ego_detection, to_persp_dir, idx)
        return ego_detection
    elif false_dets_method_str == 'random_add_remove':
        det_size = len(trust_objs)
        if det_size == 0:
            return []

        # Add a new object with set probability for each existing detection
        false_dets = []
        for i in range(0, det_size):
            if std_utils.decision_true(cfg.RANDOM_MALICIOUS_PROBABILITY):
                det_to_add = copy.deepcopy(trust_objs[i].obj)
                det_to_add.ry = np.pi * (random.random() - 0.5)
                fwd = 70 * random.random()
                det_to_add.t = (fwd * (random.random() - 0.5), det_to_add.t[1], fwd)

        new_trust_objs = []
        for obj in trust_objs:
            if std_utils.decision_true(1 - cfg.RANDOM_MALICIOUS_PROBABILITY):
                new_trust_objs.append(obj)
        trust_objs = new_trust_objs

        return false_dets
    elif false_dets_method_str == 'random_add':
        det_size = len(trust_objs)
        if det_size == 0:
            return []

        # Add a new object with set probability for each existing detection
        false_dets = []
        for i in range(0, det_size):
            if std_utils.decision_true(cfg.RANDOM_MALICIOUS_PROBABILITY):
                det_to_add = copy.deepcopy(trust_objs[i].obj)
                det_to_add.ry = np.pi * (random.random() - 0.5)
                fwd = 70 * random.random()
                det_to_add.t = (fwd * (random.random() - 0.5), det_to_add.t[1], fwd)

        return false_dets

    return []