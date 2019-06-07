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
import vehicle_trust as v_trust
import std_utils
import constants as const

# Compute and save final detections
# Only for the ego vehicle as all other vehicles are not
# guaranteed to have all nearby vehicles
def compute_final_detections():
    std_utils.delete_subdir(cfg.FINAL_DETS_SUBDIR)
    std_utils.delete_subdir(cfg.FINAL_DETS_SUBDIR_AF)

    # First for the ego vehicle
    velo_dir = cfg.DATASET_DIR + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        trust_dict = v_trust.load_vehicle_trust_objs(idx)

        perspect_trust_objs = p_utils.get_all_detections(idx, const.ego_id(), results=cfg.USE_RESULTS, filter_area=False)

        # TODO: Add fake detections

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        matching_objs = matching_utils.match_iou3ds(perspect_trust_objs, only_ego_matches=False)

        logging.debug("Matching objects!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.debug(matching_objs)
        # Aggregate messages into final detections
        final_dets = aggregate_msgs(matching_objs, trust_dict)
        logging.debug("Final detections!!!!!!!!!!!!!!!!!!!!!")
        logging.debug(final_dets)

        output_final_dets(final_dets, idx)

# Aggregates messages based on vehicle trust values, confidence, and certainty scores
def aggregate_msgs(matching_objs, trust_dict):
    final_dets = []

    for match_list in matching_objs:
        # Do not add self to the list of detections
        if match_list[0].detector_id == const.ego_id() and match_list[0].det_idx == 0:
            logging.debug("Skipping self detection")
            continue

        if len(match_list) > 1:
            match_list[0].obj.score = aggregate_score(match_list, trust_dict)
            # TODO Also average position and angles of object?
            # I don't think this is feasible when captures are off in time
            final_dets.append(match_list[0].obj)
            logging.debug("Adding multi object: {}".format(match_list[0].obj.t))
        else:
            final_dets.append(match_list[0].obj)
            logging.debug("Adding single object: {}".format(match_list[0].obj.t))

    return final_dets

def aggregate_score(match_list, trust_dict):
    if cfg.AGGREGATE_METHOD == 0:
        # Aggregate based on weighted average of scores
        count = 0
        num = 0
        den = 0
        for trust_obj in match_list:
            weight = trust_obj.detector_certainty * v_trust.vehicle_trust_value(trust_dict, trust_obj.detector_id)
            num += trust_obj.obj.score * weight
            den += weight
            count += 1

        if den == 0:
            final_score = 0
        else:
            final_score = num / (count * den)
        return final_score
    elif cfg.AGGREGATE_METHOD == 1:
        print("Test")
        # TODO Aggregate additively on weighted scores
    else:
        print("Error: Aggregation method is not properly set!!!")

def output_final_dets(objects, idx):
    filtered_objects = []
    area_filtered_objects = []

    if objects is not None and len(objects) != 0:
        # Filter detections below a low score threshold
        for obj in objects:
            if obj.score >= cfg.SCORE_THRESHOLD:
                filtered_objects.append(obj)

        # Filter for area
        area_filtered_objects = p_utils.filter_labels(filtered_objects)

    print_final_dets(filtered_objects, idx, cfg.FINAL_DETS_SUBDIR)
    print_final_dets(area_filtered_objects, idx, cfg.FINAL_DETS_SUBDIR_AF)

def print_final_dets(objects, idx, subdir):
    filepath = os.path.join(cfg.DATASET_DIR, subdir) + '/{:06d}.txt'.format(idx)
    std_utils.make_dir(filepath)

    # If no predictions, skip to next file
    if objects is None or len(objects) == 0:
        np.savetxt(filepath, [])
        return

    # Save final dets in kitti format
    # To keep each value in its appropriate position, an array of zeros
    # (N, 16) is allocated but only values [4:16] are used
    kitti_predictions = np.zeros([len(objects), 16])

    i = 0
    obj_types = []
    for obj in objects:
        # Occlusion, truncation, and alpha not used
        kitti_predictions[i, 1] = -1
        kitti_predictions[i, 2] = -1
        kitti_predictions[i, 3] = -10
        # 2D Box
        kitti_predictions[i, 4] = obj.x1
        kitti_predictions[i, 5] = obj.y1
        kitti_predictions[i, 6] = obj.x2
        kitti_predictions[i, 7] = obj.y2
        # 3D Box
        kitti_predictions[i, 8] = obj.h
        kitti_predictions[i, 9] = obj.w
        kitti_predictions[i, 10] = obj.l
        # Position
        kitti_predictions[i, 11] = obj.t[0]
        kitti_predictions[i, 12] = obj.t[1]
        kitti_predictions[i, 13] = obj.t[2]
        kitti_predictions[i, 14] = obj.ry
        kitti_predictions[i, 15] = obj.score
        i += 1

        obj_types.append(obj.type)

    # Round detections to 3 decimal places
    kitti_predictions = np.round(kitti_predictions, 3)

    # Empty Truncation, Occlusion
    kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                 dtype=np.int32)

    kitti_text = np.column_stack([obj_types,
                                  kitti_empty_1,
                                  kitti_predictions[:,3:16]])

    with open(filepath, 'w+') as f:
        np.savetxt(f, kitti_text, newline='\r\n',
            fmt='%s')