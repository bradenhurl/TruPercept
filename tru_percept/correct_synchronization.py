import os
import numpy as np
import math
import logging
import sys
from shutil import copyfile

from wavedata.tools.obj_detection import obj_utils

import tru_percept.perspective_utils as p_utils
import tru_percept.matching_utils as matching_utils
import tru_percept.config as cfg
import tru_percept.std_utils as std_utils

# Perspectives are not synchronized with the ego vehicle
# To correct this vehicle positions will be compared with ground truth from the ego vehicle
# Each detection will be matched to a GT object to obtain its speed
# If a detection is not matched its speed will be zero
# Next each object will need to be shifted
#       -> This will be done by comparing the speed/position of the detection/ego object
#          of the fastest moving vehicle to determine the time difference
# Lastly the world_position will be corrected and all detections shifted by same amount
def correct_synchro():
    std_utils.delete_all_subdirs(cfg.SYNCHRONIZED_PREDS_DIR)

    # Need to use augmented labels since they contain the speed and entity ID
    aug_label_dir = cfg.DATASET_DIR + '/label_aug_2'
    velo_dir = cfg.DATASET_DIR + '/velodyne'

    alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'
    persp_dirs = os.listdir(alt_pers_dir)

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        sys.stdout.flush()
        sys.stdout.write('\rFinished synchronization for index: {} / {}'.format(
            file_idx, num_files))
        file_idx += 1

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        logging.debug("**********************************Index: %d", idx)

        # Create dictionary for quickly obtaining speed of object
        ego_gt = obj_utils.read_labels(aug_label_dir, idx, results=False, synthetic=True)
        dict_ego_gt = {}
        for obj in ego_gt:
            dict_ego_gt[obj.id] = obj

        # Ego vehicle does not need synchronization
        # Simply copy file
        src_file = '{}/{}/{:06d}.txt'.format(cfg.DATASET_DIR, cfg.PREDICTIONS_SUBDIR, idx)
        dst_dir = '{}/{}/'.format(cfg.DATASET_DIR, cfg.SYNCHRONIZED_PREDS_DIR)
        dst_file = dst_dir + '{:06d}.txt'.format(idx)
        std_utils.make_dir(dst_dir)
        copyfile(src_file, dst_file)

        # Do for all the alternate perspectives
        for entity_str in persp_dirs:
            persp_dir = os.path.join(alt_pers_dir, entity_str)
            if not os.path.isdir(persp_dir):
                continue

            persp_det = get_synchronized_dets(persp_dir, cfg.DATASET_DIR, idx, dict_ego_gt)

            if persp_det == -1:
                continue

            # Make sure directory exists if we've made it this far
            out_dir = persp_dir + '/predictions_synchro/'
            std_utils.make_dir(out_dir)

            # If there are no detections then stop but make empty file
            # since empty file exists for predictions
            if persp_det == None:
                # Write a file with nothing as there are no detections
                with open('{}/{:06d}.txt'.format(out_dir, idx), 'w+') as f:
                    continue

            std_utils.save_objs_to_file(persp_det, idx, out_dir, True)


    print("Finished synchronizing perspectives.")

# Attempts to synchronize detections from one vehicle to the time of
# another vehicle
# Returns all detections, synchronized or not
# If gt file doesn't exist, returns -1
# if no detections, returns None
# persp_dir: Base perspective directory where detections are loaded from
# to_persp_dir: Base perspective directory detections are being synchronized with
def get_synchronized_dets(persp_dir, to_persp_dir, idx, to_persp_dict_gt=None):

    det_dir = persp_dir + '/{}'.format(cfg.PREDICTIONS_SUBDIR)
    if not os.path.isdir(det_dir):
        return -1

    gt_dir = persp_dir + '/label_aug_2'
    if not os.path.isdir(gt_dir):
        return -1

    det_filepath = persp_dir + '/{}/{:06d}.txt'.format(cfg.PREDICTIONS_SUBDIR, idx)
    if not os.path.isfile(det_filepath):
        return -1

    if to_persp_dict_gt is None:
        # Create dictionary for quickly obtaining speed of object
        to_persp_aug_label_dir = to_persp_dir + '/label_aug_2'
        to_persp_gt = obj_utils.read_labels(to_persp_aug_label_dir, idx, results=False, synthetic=True)
        to_persp_dict_gt = {}
        if to_persp_gt == None:
            return -1
        for obj in to_persp_gt:
            to_persp_dict_gt[obj.id] = obj

    gt_filepath = persp_dir + '/label_aug_2/{:06d}.txt'.format(idx)
    if not os.path.isfile(gt_filepath):
        return -1

    persp_det = obj_utils.read_labels(det_dir, idx, results=True, synthetic=False)
    persp_gt = obj_utils.read_labels(gt_dir, idx, results=False, synthetic=True)

    if persp_det == None:
        return persp_det

    # Should just match 1 to 1 with highest match
    cfg.IOU_MATCHING_THRESHOLD = 0.01

    # Match predictions to their ground truth object
    max_ious, iou_indices = matching_utils.get_iou3d_matches(persp_gt, persp_det)
    max_speed = -1
    max_speed_idx = -1
    min_ry_diff = sys.maxsize
    for obj_idx in range(0, len(iou_indices)):
        if iou_indices[obj_idx] != -1:
            matched_speed = persp_gt[int(iou_indices[obj_idx])].speed
            persp_det[obj_idx].speed = matched_speed
            persp_det[obj_idx].id = persp_gt[int(iou_indices[obj_idx])].id
            if matched_speed > max_speed:
                max_speed = matched_speed
                max_speed_idx = obj_idx

                # Should also try to take vehicle which turns the least (as it will affect speed/distance)
                ry_diff = abs(persp_gt[int(iou_indices[obj_idx])].ry - persp_det[obj_idx].ry)
                if min_ry_diff > ry_diff:
                    min_ry_diff = ry_diff
        else:
            # Object not matched so speed is unknown
            # No synchronization offset will be applied
            persp_det[obj_idx].speed = 0

    # Adjust detection positions using velocity
    # if any detection was matched with speed > 0
    if max_speed > 0:
        key = persp_det[max_speed_idx].id
        if key in to_persp_dict_gt:
            # Convert to ego vehicle coordinates
            p_utils.to_world(persp_det, persp_dir, idx)
            p_utils.to_perspective(persp_det, to_persp_dir, idx)
            p_utils.to_world(persp_gt, persp_dir, idx)
            p_utils.to_perspective(persp_gt, to_persp_dir, idx)

            # Get the time from the object with the highest speed
            obj1 = to_persp_dict_gt[key]
            obj2 = persp_det[max_speed_idx]
            pos_diff = np.asarray(obj2.t) - np.asarray(obj1.t)
            pos_diff = pos_diff.reshape((1,3))
            time = math.sqrt(np.dot(pos_diff, pos_diff.T)) / obj2.speed

            # Check if the detections are ahead or behind
            # Set the offset direction to match
            theta = np.arctan2(np.cos(obj2.ry), -np.sin(obj2.ry))
            unit_vec = np.asarray([np.cos(theta), 0, np.sin(theta)])
            offset_dir = -1
            if np.dot(pos_diff, unit_vec) > 0:
                offset_dir = 1

            # For some reason ego vehicle is opposite
            # TODO Figure out why
            if to_persp_dir == cfg.DATASET_DIR:
                offset_dir *= -1

            # Convert back to perspective coordinates then save
            p_utils.to_world(persp_det, to_persp_dir, idx)
            p_utils.to_perspective(persp_det, persp_dir, idx)

            # Adjust all the detections based on the offset time and their own speed
            for obj in persp_det:
                # First need to convert ry back to proper angle
                theta = np.arctan2(np.cos(obj.ry), -np.sin(obj.ry))
                # Next need to extract x/y components
                unit_vec = np.asarray([np.cos(theta), np.sin(theta)])

                # Lastly use the calculated time offset to create a distance
                # offset and add it to the obj position
                dist = time * obj.speed
                offset = unit_vec * dist * offset_dir
                obj.t = (obj.t[0] + offset[1], obj.t[1], obj.t[2] + offset[0])

    return persp_det