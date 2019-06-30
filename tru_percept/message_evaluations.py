import os
import sys
import numpy as np
import cv2
import logging

from wavedata.tools.obj_detection import obj_utils

import perspective_utils as p_utils
import matching_utils
import trust_utils
import config as cfg
import std_utils
import constants as const

# Compute and save message evals for each vehicle
# Files get saved to the base directory under message_evaluations
# The format is:
# Message ID, Confidence, Certainty, Evaluator ID
def compute_message_evals():
    std_utils.delete_all_subdirs(cfg.MSG_EVALS_SUBDIR)

    # First for the ego vehicle
    compute_perspect_eval(cfg.DATASET_DIR, const.ego_id())

    # Then for all the alternate perspectives
    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        compute_perspect_eval(perspect_dir, int(entity_str))
        
    print("Finished computing message evals.")

def aggregate_message_evals():
    std_utils.delete_all_subdirs(cfg.AGG_MSG_EVALS_SUBDIR)

    # First for the ego vehicle
    aggregate_persp_msg_evals(cfg.DATASET_DIR, const.ego_id())

    # Then for all the alternate perspectives
    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        aggregate_persp_msg_evals(perspect_dir, int(entity_str))

    print("Finished aggregating message evals.")


def compute_perspect_eval(perspect_dir, persp_id):
    logging.info("**********************************************************************")
    logging.info("Computing evaluations for perspective: %d", persp_id)
    velo_dir = perspect_dir + '/velodyne'
    matching_dir = perspect_dir + '/matching_test'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        logging.debug("**********************************Index: %d", idx)

        # Load predictions from own and nearby vehicles
        # First object in list will correspond to the ego_entity_id
        perspect_trust_objs = p_utils.get_all_detections(idx, persp_id, results=cfg.USE_RESULTS, filter_area=True)

        # Add fake detections to perspect_preds

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        # TODO should set this to match all, have evaluations of zero confidence with some
        # certainty for unmatched detections
        matching_objs = matching_utils.match_iou3ds(perspect_trust_objs, only_ego_matches=True)

        # Print matching objects to test with visualization
        # out_file = matching_dir + '/{:06d}.txt'.format(idx)
        # if os.path.exists(out_file):
        #     os.remove(out_file)
        # else:
        #     logging.debug("Cannot delete the file as it doesn't exists")
        # for match_list in matching_objs:
        #     if len(match_list) > 1:
        #         objs = trust_utils.strip_objs(match_list)
        #         save_filtered_objs(objs, idx, matching_dir)

        # Calculate trust from received detections
        trust_utils.get_message_trust_values(matching_objs, perspect_dir, persp_id, idx)
        save_msg_evals(matching_objs, idx)

def save_msg_evals(msg_trusts, idx):
    logging.debug("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Save msg evals in trust")
    if msg_trusts is None:
        logging.debug("Msg trusts is none")
        return

    for matched_msgs in msg_trusts:
        first_obj = True
        logging.debug("Outputting list of matched objects")
        for trust_obj in matched_msgs:
            if first_obj:
                #Skip first object as it is from self
                first_obj = False
                logging.debug("Skipping first object")
                continue

            # Fill the array to write
            msg_trust_output = np.zeros([1, 6])
            msg_trust_output[0,0] = trust_obj.det_idx
            msg_trust_output[0,1] = trust_obj.obj.score
            msg_trust_output[0,2] = trust_obj.detector_certainty
            msg_trust_output[0,3] = trust_obj.evaluator_id
            msg_trust_output[0,4] = trust_obj.evaluator_certainty
            msg_trust_output[0,5] = trust_obj.evaluator_score

            logging.debug("********************Saving trust val to id: %d at idx: %d", trust_obj.detector_id, idx)
            # Save to text file
            file_path = p_utils.get_folder(trust_obj.detector_id) + '/{}/{:06d}.txt'.format(cfg.MSG_EVALS_SUBDIR,idx)
            logging.debug("Writing msg evals to file: %s", file_path)
            std_utils.make_dir(file_path)
            with open(file_path, 'a+') as f:
                np.savetxt(f, msg_trust_output,
                           newline='\r\n', fmt='%i %f %f %i %f %f')

def load_msg_evals(persp_dir, idx):
    # Define the list
    msg_evals = []

    if idx < 0:
        return []

    filepath = persp_dir + '/' + cfg.MSG_EVALS_SUBDIR + '/{:06d}.txt'.format(idx)
    if not os.path.isfile(filepath):
        return []

    # Extract the list
    if os.stat(filepath).st_size == 0:
        return []

    p = np.loadtxt(filepath, delimiter=' ',
                   dtype=str,
                   usecols=np.arange(start=0, step=1, stop=6))

    # Check if the output is single dimensional or multi dimensional
    if len(p.shape) > 1:
        label_num = p.shape[0]
    else:
        label_num = 1

    for idx in np.arange(label_num):
        trust_obj = trust_utils.MessageEvaluation()

        if label_num > 1:
            trust_obj.det_idx = int(p[idx,0])
            trust_obj.det_score = float(p[idx,1])
            trust_obj.det_certainty = float(p[idx,2])
            trust_obj.evaluator_id = int(p[idx,3])
            trust_obj.evaluator_certainty = float(p[idx,4])
            trust_obj.evaluator_score = float(p[idx,5])
        else:
            trust_obj.det_idx = int(p[0])
            trust_obj.det_score = float(p[1])
            trust_obj.det_certainty = float(p[2])
            trust_obj.evaluator_id = int(p[3])
            trust_obj.evaluator_certainty = float(p[4])
            trust_obj.evaluator_score = float(p[5])

        msg_evals.append(trust_obj)

    return msg_evals


# Computes total msg_evals
def aggregate_persp_msg_evals(persp_dir, persp_id):
    logging.info("**********************************************************************")
    logging.info("Aggregating msg evaluations for perspective: %d", persp_id)
    velo_dir = persp_dir + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        logging.debug("**********************************Index: %d", idx)

        msg_evals = load_msg_evals(persp_dir, idx)

        eval_lists = {}
        for msg_eval in msg_evals:
            if msg_eval.det_idx in eval_lists:
                eval_lists[msg_eval.det_idx].append(msg_eval)
            else:
                eval_lists[msg_eval.det_idx] = [msg_eval]

        #print("Perspective and index: ", persp_id, idx)
        #print(eval_lists)
        eval_count = 0
        trust_sum = 0
        logging.debug(eval_lists)
        for det_idx, eval_list in eval_lists.items():
            num = 0
            den = 0
            logging.debug("det_idx: %d", det_idx)
            logging.debug("Eval list: {}".format(eval_list))
            logging.debug("Eval list len: %d", len(eval_list))
            for eval_item in eval_list:
                num += eval_item.evaluator_certainty * eval_item.evaluator_score
                den += eval_item.evaluator_certainty
                eval_count += 1
            if den == 0:
                msg_trust = 0
            else:
                msg_trust = num / den

            #TODO add option for +/- message aggregation
            save_agg_msg_eval(persp_id, idx, det_idx, msg_trust)

def save_agg_msg_eval(persp_id, idx, det_idx, msg_trust):
    logging.debug("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Save msg evals in trust")

    # Fill the array to write
    msg_trust_output = np.zeros([1, 3])
    msg_trust_output[0,0] = persp_id
    msg_trust_output[0,1] = det_idx
    msg_trust_output[0,2] = msg_trust
    #print("Saving msg agg val: ", persp_id, idx, det_idx, msg_trust)

    logging.debug("********************Saving msg trust val to id: %d at det_idx: %d for idx: %d", persp_id, det_idx, idx)
    # Save to text file
    file_path = os.path.join(os.path.join(p_utils.get_folder(persp_id), cfg.AGG_MSG_EVALS_SUBDIR), '{:06d}.txt'.format(idx))
    logging.debug("Writing msg evals to file: %s", file_path)
    std_utils.make_dir(file_path)
    print(file_path)
    with open(file_path, 'a+') as f:
        np.savetxt(f, msg_trust_output,
                   newline='\r\n', fmt='%i %i %f')


# Loads all aggregated msg evals into a dictionary
# Key 1: persp_id
# Key 2: det_idx
def load_agg_msg_evals(idx):
    # Then for all the alternate perspectives
    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

    msg_evals_dict = {}
    msg_evals_dict[const.ego_id()] = load_agg_msg_evals_from_persp(cfg.DATASET_DIR, idx)
    print("Inserting primary key: ", const.ego_id())
    for entity_str in os.listdir(alt_pers_dir):
        perspect_dir = os.path.join(alt_pers_dir, entity_str)
        if not os.path.isdir(perspect_dir):
            continue
        msg_evals_dict[int(entity_str)] = load_agg_msg_evals_from_persp(perspect_dir, idx)

        #print("Inserting primary key: ", int(entity_str))
    return msg_evals_dict

def load_agg_msg_evals_from_persp(persp_dir, idx):
    # Define the list
    msg_evals_dict = {}

    if idx < 0:
        return []

    filepath = persp_dir + '/' + cfg.AGG_MSG_EVALS_SUBDIR + '/{:06d}.txt'.format(idx)
    print(filepath)
    if not os.path.isfile(filepath):
        print("Could not find file")
        return []

    # Extract the list
    if os.stat(filepath).st_size == 0:
        print("File is empty")
        return []

    print("Loading agg from: ", filepath)
    p = np.loadtxt(filepath, delimiter=' ',
                   dtype=str,
                   usecols=np.arange(start=0, step=1, stop=3))

    # Check if the output is single dimensional or multi dimensional
    if len(p.shape) > 1:
        label_num = p.shape[0]
    else:
        label_num = 1

    for idx in np.arange(label_num):
        msg_eval = trust_utils.AggregatedMessageEvaluation()

        if label_num > 1:
            msg_eval.persp_id = int(p[idx,0])
            msg_eval.det_idx = int(p[idx,1])
            msg_eval.msg_trust = float(p[idx,2])
            msg_evals_dict[p[idx,1]] = msg_eval
        else:
            msg_eval.persp_id = int(p[0])
            msg_eval.det_idx = int(p[1])
            msg_eval.msg_trust = float(p[2])
            msg_evals_dict[p[1]] = msg_eval
        print("Inserting key: ", msg_eval.det_idx)

    return msg_evals_dict

# Function for outputting objects for visualization tests
def save_filtered_objs(gt_objs, idx, out_dir):
    out_file = out_dir + '/{:06d}.txt'.format(idx)

    with open(out_file, 'a+') as f:
        if gt_objs is None:
            return
        for obj in gt_objs:
            kitti_text_3d = '{} {} {} {} {:d} {:d} {:d} {:d} {} {} {} {} {} {} {}'.format(obj.type,
                obj.truncation, obj.occlusion, obj.alpha, int(obj.x1), int(obj.y1), int(obj.x2),
                int(obj.y2), obj.h, obj.w, obj.l, obj.t[0], obj.t[1], obj.t[2], obj.ry)

            f.write('%s\n' % kitti_text_3d)