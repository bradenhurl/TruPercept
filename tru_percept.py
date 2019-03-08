import cv2
import numpy as np
import os
import sys
from wavedata.tools.obj_detection import obj_utils
import perspective_utils as p_utils
import matching_utils
import trust_utils

# Change this folder to point to the Trust Perception dataset
dataset_dir = os.path.expanduser('~') + '/wavedata-dev/demos/gta/training/'

def main():
    
    velo_dir = dataset_dir + 'velodyne'
    calib_dir = dataset_dir + 'calib'
    predictions_dir = dataset_dir + 'predictions'
    output_dir = dataset_dir + 'predictions_tru_percept'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        # Load predictions from own vehicle
        #TODO Test if certainty values are corresponding correctly
        ego_preds = obj_utils.read_labels(predictions_dir, idx, results=True)
        ego_trust_objs = trust_utils.createTrustObjects(dataset_dir, idx, trust_utils.self_id, ego_preds)

        # Load predictions from nearby vehicles
        perspect_preds = p_utils.get_all_detections(dataset_dir, idx, results=True)

        # Add fake detections to perspect_preds

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        matching_objs = matching_utils.match_iou3ds(ego_preds, perspect_preds)

        # Calculate trust from received detections

        # Adjust predictions

        # Add to vehicle trust

        print("Index: ", idx)
        print("Ego preds: ", ego_preds)
        print("perspect_preds: ", perspect_preds)

        sys.stdout.write("\rWorking on idx: {} / {}".format(
                file_idx + 1, num_files))
        sys.stdout.flush()
        file_idx = file_idx + 1


main()