import os
import numpy as np
import math
import logging

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

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue
        logging.debug("**********************************Index: %d", idx)

        # Create dictionary for quickly obtaining speed of object
        ego_gt = obj_utils.read_labels(aug_label_dir, idx, results=False, synthetic=True)
        dict_ego_gt = {}
        for obj in ego_gt:
            dict_ego_gt[obj.id] = obj

        # Ego vehicle does not need synchronization
        # Do for all the alternate perspectives
        for entity_str in persp_dirs:
            perspect_dir = os.path.join(alt_pers_dir, entity_str)
            if not os.path.isdir(perspect_dir):
                continue

            det_dir = perspect_dir + '/{}'.format(cfg.PREDICTIONS_SUBDIR)
            if not os.path.isdir(det_dir):
                continue

            gt_dir = perspect_dir + '/label_aug_2'
            if not os.path.isdir(gt_dir):
                continue

            det_filepath = perspect_dir + '/{}/{:06d}.txt'.format(cfg.PREDICTIONS_SUBDIR, idx)
            if not os.path.isfile(det_filepath):
                continue

            gt_filepath = perspect_dir + '/label_aug_2/{:06d}.txt'.format(idx)
            if not os.path.isfile(gt_filepath):
                continue

            persp_det = obj_utils.read_labels(det_dir, idx, results=True, synthetic=False)
            persp_gt = obj_utils.read_labels(gt_dir, idx, results=False, synthetic=True)

            if persp_det == None:
                continue

            # Should just match 1 to 1 with highest match
            cfg.IOU_MATCHING_THRESHOLD = 0.5

            # Match predictions to their ground truth object
            max_ious, iou_indices = matching_utils.get_iou3d_matches(persp_gt, persp_det)
            max_speed = -1
            max_speed_idx = -1
            print(iou_indices)
            for obj_idx in range(0, len(iou_indices)):
                if iou_indices[obj_idx] != -1:
                    matched_speed = persp_gt[int(iou_indices[obj_idx])].speed
                    persp_det[obj_idx].speed = matched_speed
                    persp_det[obj_idx].id = persp_gt[int(iou_indices[obj_idx])].id
                    if matched_speed > max_speed:
                        max_speed = matched_speed
                        max_speed_idx = obj_idx

            print(entity_str)

            # Adjust detection positions using velocity
            # if any detection was matched with speed > 0
            if max_speed > 0:
                key = persp_det[max_speed_idx].id
                if key in dict_ego_gt:
                    # Convert to ego vehicle coordinates
                    p_utils.to_world(persp_det, perspect_dir, idx)
                    p_utils.to_perspective(persp_det, cfg.DATASET_DIR, idx)
                    p_utils.to_world(persp_gt, perspect_dir, idx)
                    p_utils.to_perspective(persp_gt, cfg.DATASET_DIR, idx)

                    # Get the time from the object with the highest speed
                    obj1 = dict_ego_gt[key]
                    obj2 = persp_det[max_speed_idx]
                    pos_diff = np.asarray(obj2.t) - np.asarray(obj1.t)
                    pos_diff = pos_diff.reshape((1,3))
                    time = math.sqrt(np.dot(pos_diff, pos_diff.T)) / obj2.speed

                    # Adjust all the detections based on the offset time and their own speed
                    for obj in persp_det:
                        unit_vec = np.asarray([np.sin(obj.ry), np.cos(obj.ry)])
                        print(unit_vec)
                        dist = time * obj.speed
                        offset = unit_vec * -dist
                        obj.t = (obj.t[0] + offset[1], obj.t[1], obj.t[2] + offset[0])

                    # Convert back to perspective coordinates then save
                    p_utils.to_world(persp_det, cfg.DATASET_DIR, idx)
                    p_utils.to_perspective(persp_det, perspect_dir, idx)

            out_dir = perspect_dir + '/predictions_synchro/'
            print("Saving to dir: ", out_dir)
            std_utils.make_dir(out_dir)
            std_utils.save_objs_to_file(persp_det, idx, out_dir, True)


    print("Finished synchronizing perspectives.")

correct_synchro()