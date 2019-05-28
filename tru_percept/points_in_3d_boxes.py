import os
import shutil
import sys
import numpy as np
import cv2
import logging

from wavedata.tools.obj_detection import obj_utils

import perspective_utils as p_utils
import trust_utils
import config as cfg
import std_utils
import certainty_utils

# Compute and save # of points for own and received detections for each vehicle
# Labels filtered for area before computing
# Files get saved to the perspective directory under cfg.POINTS_IN_3D_BOXES_DIR
# The format is:
# Detector ID, Detection ID, # points in 3D box
# Where Detector ID is the ID of the vehicle which detected the object
# Detection ID is the index of the detection from the Detector ID vehicle
def compute_points_in_3d_boxes():

    print("Beginning calculation of points_in_3d_boxes")
    
    # Obtain the ego ID
    ego_folder = cfg.DATASET_DIR + '/ego_object'
    ego_info = obj_utils.read_labels(ego_folder, 0, synthetic=True)
    ego_id = ego_info[0].id
    
    std_utils.delete_all_subdirs(cfg.POINTS_IN_3D_BOXES_DIR)

    # First for the ego vehicle
    compute_perspect_points_in_3d_boxes(cfg.DATASET_DIR, ego_id, ego_id)

    # Then for all the alternate perspectives
    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

    persp_count = len(os.listdir(alt_pers_dir))
    persp_idx = 0
    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        compute_perspect_points_in_3d_boxes(perspect_dir, int(entity_str), ego_id)

        sys.stdout.flush()
        sys.stdout.write('\rFinished point count for perspective {}: {} / {}'.format(
            int(entity_str), persp_idx, persp_count))
        persp_idx += 1


def compute_perspect_points_in_3d_boxes(perspect_dir, persp_id, ego_id):
    logging.info("**********************************************************************")
    logging.info("Computing points_in_3d_boxes for perspective: %d", persp_id)
    velo_dir = perspect_dir + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        logging.debug("**********************************Index: %d", idx)

        logging.critical("Need to get ego detections from whichever entity_id we're on")

        # Load predictions from own and nearby vehicles
        # First object in list will correspond to the ego_entity_id
        trust_objs = p_utils.get_all_detections(ego_id, idx, persp_id, results=False, filter_area=False)

        save_points_in_3d_boxes(trust_objs, ego_id, idx, perspect_dir, persp_id)

        sys.stdout.flush()
        sys.stdout.write('\rFinished point count for file index: {} / {}'.format(
            file_idx, num_files))
        file_idx += 1

    if file_idx > 0:
        print("\nFinished non-null perspective: ", int(persp_id))

def save_points_in_3d_boxes(trust_objs, ego_id, idx, perspect_dir, persp_id):
    if trust_objs is None:
        logging.debug("trust_objs is none")
        return

    det_count = 0
    for obj_list in trust_objs:
        det_count += len(obj_list)

    logging.debug("********************Saving points_in_3d_boxes val to id: {} at idx: {}".format(persp_id, idx))
    # Save to text file
    file_path = p_utils.get_folder(ego_id, persp_id) + '/{}/{:06d}.txt'.format(cfg.POINTS_IN_3D_BOXES_DIR,idx)
    std_utils.make_dir(file_path)
    logging.debug("Writing points_in_3d_boxes to file: %s", file_path)

    with open(file_path, 'a+') as f:
        for obj_list in trust_objs:
            for trust_obj in obj_list:
                pc = certainty_utils.get_nan_point_cloud(perspect_dir, idx)
                num_points = certainty_utils.numPointsIn3DBox(trust_obj.obj, pc, perspect_dir, idx)

                # Fill the array to write
                output = np.zeros([1, 3])
                output[0,0] = trust_obj.detector_id
                output[0,1] = trust_obj.det_idx
                output[0,2] = num_points

                np.savetxt(f, output, newline='\r\n', fmt='%i %i %i')

compute_points_in_3d_boxes()