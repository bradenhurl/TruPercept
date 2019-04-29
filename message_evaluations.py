import cv2
import numpy as np
import os
import sys
from wavedata.tools.obj_detection import obj_utils
import perspective_utils as p_utils
import matching_utils
import trust_utils

#TODO - Create this as a function and pass in the dataset_dir
# Change this folder to point to the Trust Perception dataset
dataset_dir = os.path.expanduser('~') + '/wavedata-dev/demos/gta/training/'

# Compute and save message evals for each vehicle
# Files get saved to the base directory under message_evaluations
# The format is:
# Message ID, Confidence, Certainty, Evaluator ID
def compute_message_evals(base_dir):

    # Obtain the ego ID
    ego_folder = base_dir + '/ego_object'
    ego_info = obj_utils.read_labels(ego_folder, 0, synthetic=True)
    ego_id = ego_info[0].id
    
    trust_utils.delete_msg_evals(base_dir)

    # First for the ego vehicle
    compute_perspect_eval(base_dir, base_dir, ego_id, ego_id)

    # Then for all the alternate perspectives
    alt_pers_dir = base_dir + '/alt_perspective/'

    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        compute_perspect_eval(base_dir, perspect_dir, int(entity_str), ego_id)


def compute_perspect_eval(base_dir, perspect_dir, persp_id, ego_id):
    print("**********************************************************************")
    print("Computing evaluations for perspective: ", persp_id)
    velo_dir = perspect_dir + '/velodyne'
    calib_dir = perspect_dir + '/calib'
    predictions_dir = perspect_dir + '/predictions'
    output_dir = perspect_dir + '/predictions_tru_percept'
    matching_dir = perspect_dir + '/matching_test'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        #TODO Testing 3 indices remove this when finished testing
        if idx < 7 or idx > 10:
            continue
        print("**********************************Index: ", idx)

        print("Need to get ego detections from whichever entity_id we're on")

        # Load predictions from own and nearby vehicles
        # First object in list will correspond to the ego_entity_id
        perspect_trust_objs = p_utils.get_all_detections(dataset_dir, ego_id, idx, persp_id, results=False, filter_area=True)

        # Add fake detections to perspect_preds

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        matching_objs = matching_utils.match_iou3ds(perspect_trust_objs, False)

        # Print matching objects to test with visualization
        # out_file = matching_dir + '/{:06d}.txt'.format(idx)
        # if os.path.exists(out_file):
        #     os.remove(out_file)
        # else:
        #     print("Can not delete the file as it doesn't exists")
        # for match_list in matching_objs:
        #     if len(match_list) > 1:
        #         objs = trust_utils.strip_objs(match_list)
        #         save_filtered_objs(objs, idx, matching_dir)

        # Calculate trust from received detections
        trust_utils.get_message_trust_values(matching_objs, perspect_dir, idx)
        trust_utils.save_msg_evals(matching_objs, base_dir, ego_id, idx)



compute_message_evals(dataset_dir)