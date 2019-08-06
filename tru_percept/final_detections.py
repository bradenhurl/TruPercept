import os
import sys
import numpy as np
import cv2
import logging

from wavedata.tools.obj_detection import obj_utils

import config as cfg
import perspective_utils as p_utils
import matching_utils
import trust_utils
import vehicle_trust as v_trust
import std_utils
import constants as const
import message_evaluations as msg_evals
import plausibility_checker
from tools.visualization import vis_matches
from tools.visualization import vis_objects

# Compute and save final detections
# Only for the ego vehicle as all other vehicles are not
# guaranteed to have all nearby vehicles
def compute_final_detections():
    print("Aggregate method: ", cfg.AGGREGATE_METHOD)
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

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        matching_objs = matching_utils.match_iou3ds(perspect_trust_objs, only_ego_matches=False)

        logging.debug("Matching objects!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.debug(matching_objs)
        # Aggregate messages into final detections
        final_dets = aggregate_msgs(matching_objs, trust_dict, idx)
        logging.debug("Final detections!!!!!!!!!!!!!!!!!!!!!")
        logging.debug(final_dets)

        output_final_dets(final_dets, idx)

    print("Finished computing final detections")

# Aggregates messages based on vehicle trust values, confidence, and certainty scores
def aggregate_msgs(matching_objs, trust_dict, idx):
    final_dets = []

    msg_evals_dict = msg_evals.load_agg_msg_evals(idx)

    if cfg.VISUALIZE_AGG_EVALS:
        for match_list in matching_objs:
            for trust_obj in match_list:
                if trust_obj.detector_id in msg_evals_dict:
                    if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                        trust_obj.obj.score = msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]
                        print("Setting trust_obj score to: ", trust_obj.obj.score)
        # print(matching_objs[0][0].obj.score)
        vis_matches.visualize_matches(matching_objs, idx, cfg.USE_RESULTS,
                                        False, -1, vis_eval_scores=True)

    for match_list in matching_objs:
        # Do not add self to the list of detections
        if match_list[0].detector_id == const.ego_id() and match_list[0].det_idx == 0:
            logging.debug("Skipping self detection")
            continue

        match_list[0].obj.score = aggregate_score(match_list, trust_dict, idx, msg_evals_dict)
        final_dets.append(match_list[0].obj)
        logging.debug("Adding multi object: {}".format(match_list[0].obj.t))

    if cfg.VISUALIZE_FINAL_DETS:
        vis_objects.visualize_objects(final_dets, idx, cfg.USE_RESULTS, False, -1, vis_scores=True)

    return final_dets

# Would be good to experiment with average position and angles of object?
# I don't think this is feasible when captures are off in time
def aggregate_score(match_list, trust_dict, idx, msg_evals_dict):

    final_score = 0.0

    # TODO potentially add local threshold to simply believe local
    # detections with a high score
    # if cfg.LOCAL_THRESHOLD <= 1 and //
    #     match_list[0].detector_id == const.ego_id() and //
    #     match_list[0].obj.score >= cfg.LOCAL_THRESHOLD:
    #     final_score += match_list[0].obj.score

    # Aggregate based on weighted average of scores
    if cfg.AGGREGATE_METHOD == 0:
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

    # Aggregate additively on weighted scores
    elif cfg.AGGREGATE_METHOD == 1:
        for trust_obj in match_list:
            weight = trust_obj.detector_certainty * v_trust.vehicle_trust_value(trust_dict, trust_obj.detector_id)
            final_score += trust_obj.obj.score * weight

    # TruPercept 1
    # Aggregate based on overall message evaluations
    elif cfg.AGGREGATE_METHOD == 2:
        den = 0
        num = 0.0
        for trust_obj in match_list:
            found = False
            if trust_obj.detector_id in msg_evals_dict:
                if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                    num += msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]
                    found = True

            if not found and trust_obj.detector_id == const.ego_id():
                num += trust_obj.obj.score

            den += 1

        if den == 0:
            final_score = 0
        else:
            final_score = num / den

    # BA 1
    elif cfg.AGGREGATE_METHOD == 3:
        final_score = 1.0

    # BA 2
    elif cfg.AGGREGATE_METHOD == 4:
        if len(match_list) > 1:
            final_score = 1.0
        else:
            final_score = match_list[0].obj.score

    # BA 3
    elif cfg.AGGREGATE_METHOD == 5:
        if len(match_list) > 1:
            final_score = 1.0
        elif match_list[0].detector_id == const.ego_id():
            final_score = match_list[0].obj.score

    # BA 4 - This one seems to work the best
    elif cfg.AGGREGATE_METHOD == 6:
        if match_list[0].detector_id == const.ego_id():
            if len(match_list) > 1:
                final_score = 1.0
            else:
                final_score = match_list[0].obj.score

    elif cfg.AGGREGATE_METHOD == 7:
        if match_list[0].detector_id == const.ego_id():
            final_score = match_list[0].obj.score

        # Check to ensure ego-vehicle is not matching with its own detections
        first = True
        for obj in match_list:
            if first:
                first = False
            elif obj.detector_id == const.ego_id():
                print("Ego objects matched!!!!")

    # Aggregate additively
    # Ego vehicle gets weight of 1
    # Other vehicles weighted at 0.5
    elif cfg.AGGREGATE_METHOD == 8:
        if match_list[0].detector_id == const.ego_id():
            final_score = match_list[0].obj.score

        for trust_obj in match_list:
            final_score += trust_obj.obj.score

        final_score = final_score / 2

    # Aggregate based on overall message evaluations
    # Same as 2 but average msg evals with ego vehicle confidence
    elif cfg.AGGREGATE_METHOD == 9:
        den = 0.0
        num = 0.0
        for trust_obj in match_list:
            found = False
            if trust_obj.detector_id in msg_evals_dict:
                if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                    num += msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]
                    found = True

            if not found and trust_obj.detector_id == const.ego_id():
                num += trust_obj.obj.score

            den += 1.0

        if den == 0:
            final_score = 0
        else:
            final_score = num / den

        # Bias the detection towards the local detection score
        if match_list[0].detector_id == const.ego_id():
            final_score += match_list[0].obj.score
            final_score /= 2

        # No need to plausibility check ego-vehicle detections or null detections
        if match_list[0].detector_id != const.ego_id() and final_score > 0:
            if not plausibility_checker.is_plausible(match_list[0].obj, idx, match_list[0].detector_id, match_list[0].det_idx):
                final_score = 0.0

    # TruPercept 2
    elif cfg.AGGREGATE_METHOD == 10:
        den = 0.0
        num = 0.0
        for trust_obj in match_list:
            found = False
            if trust_obj.detector_id in msg_evals_dict:
                if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                    num += msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]
                    found = True

            if not found:
                num += trust_obj.obj.score

            den += 1.0

        if den == 0:
            final_score = 0.0
        else:
            final_score = num / den

        # No need to plausibility check ego-vehicle detections or null detections
        if match_list[0].detector_id != const.ego_id() and final_score > 0:
            if not plausibility_checker.is_plausible(match_list[0].obj, idx, match_list[0].detector_id, match_list[0].det_idx):
                final_score = 0.0

    # Only use ego vehicle detections but adjust position
    elif cfg.AGGREGATE_METHOD == 11:
        # No need to plausibility check ego-vehicle detections or null detections
        if match_list[0].detector_id == const.ego_id():
            final_score = match_list[0].obj.score
            min_dist = sys.float_info.max
            for trust_obj in match_list:
                if trust_obj.detector_dist < min_dist:
                    min_dist = trust_obj.detector_dist
                    match_list[0].t = trust_obj.obj.t
                    match_list[0].ry = trust_obj.obj.ry

    # TruPercept 3
    # Ego-vehicle if visible in range, otherwise use trupercept
    elif cfg.AGGREGATE_METHOD == 12:
        trust_obj = match_list[0]
        # No need to plausibility check ego-vehicle detections or null detections
        if trust_obj.detector_id == const.ego_id():
            final_score = trust_obj.obj.score
        else:
            #Check if in range
            obj_pos = np.asanyarray(trust_obj.obj.t)
            obj_dist = np.sqrt(np.dot(obj_pos, obj_pos.T))
            if obj_dist < cfg.MAX_LIDAR_DIST:
                # exclude if >= 10 points in box (Should detect if visible)
                if trust_obj.evaluator_3d_points < 10:
                    # if not many points in box then add if it is plausible
                    if plausibility_checker.is_plausible(match_list[0].obj, idx, match_list[0].detector_id, match_list[0].det_idx):
                        final_score = trust_obj.obj.score
            else:
                if trust_obj.detector_id in msg_evals_dict:
                    if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                        final_score = msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]

    # Ego-vehicle if visible in range, otherwise use trupercept. With position corrections.
    elif cfg.AGGREGATE_METHOD == 13:
        trust_obj = match_list[0]
        # No need to plausibility check ego-vehicle detections or null detections
        if trust_obj.detector_id == const.ego_id():
            final_score = trust_obj.obj.score
        else:
            #Check if in range
            obj_pos = np.asanyarray(trust_obj.obj.t)
            obj_dist = np.sqrt(np.dot(obj_pos, obj_pos.T))
            if obj_dist < cfg.MAX_LIDAR_DIST:
                # exclude if >= 10 points in box (Should detect if visible)
                if trust_obj.evaluator_3d_points < 10:
                    # if not many points in box then add if it is plausible
                    if plausibility_checker.is_plausible(match_list[0].obj, idx, match_list[0].detector_id, match_list[0].det_idx):
                        final_score = trust_obj.obj.score
            else:
                if trust_obj.detector_id in msg_evals_dict:
                    if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                        final_score = msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]

        if final_score > 0.0:
            min_dist = sys.float_info.max
            for obj in match_list:
                if obj.detector_dist < min_dist:
                    min_dist = obj.detector_dist
                    match_list[0].t = obj.obj.t
                    match_list[0].ry = obj.obj.ry

    else:
        print("Error: Aggregation method is not properly set!!!")

    # Ensure final_score is within proper range
    final_score = min(final_score, 1.0)
    final_score = max(final_score, 0.0)
    
    logging.debug("Final detection aggregation. Idx: {}  pos: {}".format(idx,match_list[0].obj.t))
    for trust_obj in match_list:
        eval_dict_score = 0.0
        if trust_obj.detector_id in msg_evals_dict:
            if trust_obj.det_idx in msg_evals_dict[trust_obj.detector_id]:
                eval_dict_score = msg_evals_dict[trust_obj.detector_id][trust_obj.det_idx]

        logging.debug("Rec detection det_id: {}, det_idx: {} score: {}, certainty: {}, msg_eval_dict: {}".format(
            trust_obj.detector_id, trust_obj.det_idx,
            trust_obj.obj.score, trust_obj.detector_certainty,
            eval_dict_score))

    logging.debug("Final score: {}".format(final_score))
    return final_score

def output_final_dets(objects, idx):
    filtered_objects = []
    area_filtered_objects = []

    if objects is not None and len(objects) != 0:
        # Filter detections below a low score threshold
        for obj in objects:
            if obj.score >= cfg.SCORE_THRESHOLD:
                filtered_objects.append(obj)

        #filtered_objects = p_utils.filter_labels(filtered_objects, False)

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