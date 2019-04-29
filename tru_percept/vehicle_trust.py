import os
import shutil

import message_evaluations as msg_evals

def calculate_vehicle_trusts(base_dir):

    # Obtain the ego ID
    ego_folder = base_dir + '/ego_object'
    ego_info = obj_utils.read_labels(ego_folder, 0, synthetic=True)
    ego_id = ego_info[0].id

    # Before calculating, first delete all previous vehicle trust values
    delete_vehicle_trust_values(base_dir)

    velo_dir = base_dir + '/velodyne'
    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        last_trust_values = load_trust_values(base_dir, idx - 1)

        # First for the ego vehicle
        compute_vehicle_trust(base_dir, base_dir, ego_id, ego_id, idx, last_trust_values)

        # Then for all the alternate perspectives
        alt_pers_dir = base_dir + '/alt_perspective/'

        for entity_str in os.listdir(alt_pers_dir):
            perspect_dir = os.path.join(alt_pers_dir, entity_str)
            if not os.path.isdir(perspect_dir):
                continue
            compute_vehicle_trust(base_dir, perspect_dir, int(entity_str), ego_id, idx, last_trust_values)

def compute_vehicle_trust(base_dir, persp_dir, persp_id, ego_id, idx, last_trust_values):


############################################################################################
# Utility functions

# Returns a dictionary with the vehicle trust values from the given index
def load_trust_values(base_dir, idx):
    if idx < 0:
        return {}

    filepath = base_dir + '/' + V_TRUST_SUBDIR + '/{:06d}.txt'.format(idx)
    #TODO return dictionary from text document

def vehicle_trust_value(trust_values, v_id):
    if v_id in trust_values:
        return trust_values[v_id]
    else:
        return DEFAULT_VEHICLE_TRUST_VAL

calculate_vehicle_trusts(dataset_dir)