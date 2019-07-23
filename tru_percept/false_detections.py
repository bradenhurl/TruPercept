import numpy as np
import os
import math

from wavedata.tools.obj_detection import obj_utils

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
                    to_persp_dir, dataset_dir):

    if int(persp_id) not in false_dets_dict:
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

    return []