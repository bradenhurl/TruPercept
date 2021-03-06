import os
import shutil
import numpy as np
import logging

from wavedata.tools.obj_detection import obj_utils

import config as cfg
import trust_utils
import std_utils
import constants as const
import message_evaluations
import perspective_utils as p_utils

def calculate_vehicle_trusts():

    # Before calculating, first delete all previous vehicle trust values
    std_utils.delete_subdir(cfg.V_TRUST_SUBDIR)

    # Initialize dictionary for vehicle trust values
    # Entity ID/VehicleTrust object pairs
    trust_dict = {}

    velo_dir = cfg.DATASET_DIR + '/velodyne'
    velo_files = os.listdir(velo_dir)
    for idx in range(cfg.MIN_IDX, cfg.MAX_IDX + 1):
        filepath = velo_dir + '/{:06d}.bin'.format(idx)

        if not os.path.isfile(filepath):
            logging.debug("Could not find file: %s", filepath)
            logging.debug("Stopping at idx: %d", idx)
            break

        # Load stale trust dict if we need it (past msg fresh period)
        stale_trust_dict = {}
        if (idx - cfg.STALE_EVALS_TIME) >= 0:
            stale_trust_dict = load_vehicle_trust_objs(idx - cfg.STALE_EVALS_TIME)

        # First for the ego vehicle
        compute_vehicle_trust(cfg.DATASET_DIR, const.ego_id(), idx, trust_dict, stale_trust_dict)

        # Then for all the alternate perspectives
        for entity_str in const.valid_perspectives():
            perspect_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)
            compute_vehicle_trust(perspect_dir, int(entity_str), idx, trust_dict, stale_trust_dict)

        write_trust_vals(trust_dict, idx)

    print("Finished calculating vehicle trusts")

def compute_vehicle_trust(persp_dir, persp_id, idx, trust_dict, stale_trust_dict):
    msg_evals = message_evaluations.load_msg_evals(persp_dir, idx)
    detections = p_utils.get_detections(persp_dir, persp_dir, idx,
                                        persp_id, persp_id, results=cfg.USE_RESULTS)

    eval_lists = {}
    for msg_eval in msg_evals:
        if msg_eval.det_idx in eval_lists:
            eval_lists[msg_eval.det_idx].append(msg_eval)
        else:
            eval_lists[msg_eval.det_idx] = [msg_eval]

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
            # Don't use own evaluations for vehicle trust
            if eval_item.evaluator_id != persp_id:
                num += eval_item.evaluator_certainty * eval_item.evaluator_score
                den += eval_item.evaluator_certainty
        if den == 0:
            msg_trust = 0
        else:
            msg_trust = num / den

        msg_trust = max(msg_trust, -1.0)
        msg_trust = min(msg_trust, 1.0)

        trust_sum += msg_trust * detections[det_idx].obj.score
        eval_count += detections[det_idx].obj.score

    # Obtain VehicleTrust object, create new object if new vehicle
    print_test = False
    if persp_id in trust_dict:
        v_trust = trust_dict[persp_id]
    else:
        v_trust = trust_utils.VehicleTrust()
        logging.debug("New trust object")
        logging.debug("v_trust value: %f", v_trust.val)
        trust_dict[persp_id] = v_trust
        print_test = True

    # Update trust with current evaluations
    v_trust.sum += trust_sum
    v_trust.count += eval_count
    v_trust.curr_score = trust_sum
    v_trust.curr_count = eval_count

    # Remove stale evaluations
    if (idx - cfg.STALE_EVALS_TIME) >= 0:
        if persp_id in stale_trust_dict:
            v_trust.sum -= stale_trust_dict[persp_id].curr_score
            v_trust.count -= stale_trust_dict[persp_id].curr_count

    if v_trust.count > 0:
        v_trust.val = v_trust.sum / v_trust.count
    else:
        v_trust.val = cfg.DEFAULT_VEHICLE_TRUST_VAL

    if print_test:
        logging.debug("test value: %f", v_trust.val)
        logging.debug("map value: %f", trust_dict[persp_id].val)


############################################################################################
# Utility functions

# Returns a dictionary with the vehicle trust values from the given index
def load_vehicle_trust_objs(idx):
    # Define the dictionary
    v_trust_dict = {}

    if idx < 0:
        return {}

    filepath = cfg.DATASET_DIR + '/' + cfg.V_TRUST_SUBDIR + '/{:06d}.txt'.format(idx)
    if not os.path.isfile(filepath):
        print("Could not find vehicle trust filepath: ", filepath)
        return {}

    # Extract the list
    if os.stat(filepath).st_size == 0:
        print("Filesize 0 for vehicle trust filepath: ", filepath)
        return {}

    p = np.loadtxt(filepath, delimiter=' ',
                   dtype=str,
                   usecols=np.arange(start=0, step=1, stop=6))

    # Check if the output is single dimensional or multi dimensional
    if len(p.shape) > 1:
        label_num = p.shape[0]
    else:
        label_num = 1

    for idx in np.arange(label_num):
        trust_obj = trust_utils.VehicleTrust()

        if label_num > 1:
            trust_obj.val = float(p[idx,1])
            trust_obj.sum = float(p[idx,2])
            trust_obj.count = int(p[idx,3])
            trust_obj.curr_score = float(p[idx,4])
            trust_obj.curr_count = int(p[idx,5])
            v_trust_dict[int(p[idx,0])] = trust_obj
        else:
            trust_obj.val = float(p[1])
            trust_obj.sum = float(p[2])
            trust_obj.count = int(p[3])
            trust_obj.curr_score = float(p[4])
            trust_obj.curr_count = int(p[5])
            v_trust_dict[int(p[0])] = trust_obj

    return v_trust_dict

def vehicle_trust_value(trust_values, v_id):
    if v_id in trust_values:
        return trust_values[v_id].val
    else:
        return cfg.DEFAULT_VEHICLE_TRUST_VAL

def write_trust_vals(trust_dict, idx):
    trust_vals_array = np.zeros([len(trust_dict), 6])
    v_count = 0
    for entity_id, trust_obj in trust_dict.items():
        trust_vals_array[v_count,0] = entity_id
        trust_vals_array[v_count,1] = max(0., trust_obj.val)
        trust_vals_array[v_count,2] = trust_obj.sum
        trust_vals_array[v_count,3] = trust_obj.count
        trust_vals_array[v_count,4] = trust_obj.curr_score
        trust_vals_array[v_count,5] = trust_obj.curr_count
        v_count += 1

    filepath = cfg.DATASET_DIR + '/' + cfg.V_TRUST_SUBDIR + '/{:06d}.txt'.format(idx)
    std_utils.make_dir(filepath)
    with open(filepath, 'w+') as f:
        np.savetxt(f, trust_vals_array,
               newline='\r\n', fmt='%i %f %f %i %f %i')