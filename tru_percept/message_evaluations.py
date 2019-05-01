import os
import shutil
import sys
import numpy as np
import cv2

from wavedata.tools.obj_detection import obj_utils

import perspective_utils as p_utils
import matching_utils
import trust_utils
import config as cfg

# Compute and save message evals for each vehicle
# Files get saved to the base directory under message_evaluations
# The format is:
# Message ID, Confidence, Certainty, Evaluator ID
def compute_message_evals():

    # Obtain the ego ID
    ego_folder = cfg.DATASET_DIR + '/ego_object'
    ego_info = obj_utils.read_labels(ego_folder, 0, synthetic=True)
    ego_id = ego_info[0].id
    
    delete_msg_evals()

    # First for the ego vehicle
    compute_perspect_eval(cfg.DATASET_DIR, ego_id, ego_id)

    # Then for all the alternate perspectives
    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        compute_perspect_eval(perspect_dir, int(entity_str), ego_id)


def compute_perspect_eval(perspect_dir, persp_id, ego_id):
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

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        print("**********************************Index: ", idx)

        print("Need to get ego detections from whichever entity_id we're on")

        # Load predictions from own and nearby vehicles
        # First object in list will correspond to the ego_entity_id
        perspect_trust_objs = p_utils.get_all_detections(ego_id, idx, persp_id, results=False, filter_area=True)

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
        trust_utils.get_message_trust_values(matching_objs, perspect_dir, persp_id, idx)
        save_msg_evals(matching_objs, ego_id, idx)

def save_msg_evals(msg_trusts, ego_id, idx):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Save msg evals in trust")
    if msg_trusts is None:
        print("Msg trusts is none")
        return

    for matched_msgs in msg_trusts:
        first_obj = True
        print("Outputting list of matched objects")
        for trust_obj in matched_msgs:
            if first_obj:
                #Skip first object as it is from self
                first_obj = False
                print("Skipping first object")
                continue

            # Fill the array to write
            msg_trust_output = np.zeros([1, 6])
            #TODO - fill in correct values
            msg_trust_output[0,0] = trust_obj.det_idx
            msg_trust_output[0,1] = trust_obj.obj.score
            msg_trust_output[0,2] = trust_obj.detector_certainty
            msg_trust_output[0,3] = trust_obj.evaluator_id
            msg_trust_output[0,4] = trust_obj.evaluator_certainty
            msg_trust_output[0,5] = trust_obj.evaluator_score

            print("********************Saving trust val to id: ", trust_obj.detector_id, " at idx: ", idx)
            # Save to text file
            file_path = p_utils.get_folder(ego_id, trust_obj.detector_id) + '/{}/{:06d}.txt'.format(cfg.MSG_EVALS_SUBDIR,idx)
            print("Writing msg evals to file: ", file_path)
            make_dir(file_path)
            with open(file_path, 'a+') as f:
                np.savetxt(f, msg_trust_output,
                           newline='\r\n', fmt='%i %f %f %i %f %f')

def make_dir(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def delete_msg_evals():
    dirpath = os.path.join(cfg.DATASET_DIR, cfg.MSG_EVALS_SUBDIR)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    altPerspect_dir = cfg.DATASET_DIR + '/alt_perspective/'
    for entity_str in os.listdir(altPerspect_dir):
        perspect_dir = os.path.join(altPerspect_dir, entity_str)
        dirpath = os.path.join(perspect_dir, cfg.MSG_EVALS_SUBDIR)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            print("Deleting directory: ", dirpath)
            shutil.rmtree(dirpath)


compute_message_evals()