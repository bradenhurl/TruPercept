import numpy as np
import logging

from wavedata.tools.obj_detection import obj_utils, evaluation
from avod.core import box_3d_encoder

import trust_utils
import tru_percept.config as cfg

# Returns indices of objects
# This function is modified code from: https://github.com/kujason
def get_iou3d_matches(ego_objs, objs_perspectives):
    all_3d_ious = []

    if len(ego_objs) > 0 and \
                objs_perspectives is not None and len(objs_perspectives) > 0:

        ego_objs_boxes_3d = [box_3d_encoder.object_label_to_box_3d(ego_obj)
                           for ego_obj in ego_objs]
        perspect_objs_boxes_3d = [box_3d_encoder.object_label_to_box_3d(objs_perspective)
                         for objs_perspective in objs_perspectives]

        # Convert to iou format
        ego_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(ego_objs_boxes_3d)
        perspect_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(perspect_objs_boxes_3d)

        max_ious_3d = np.zeros(len(objs_perspectives))
        max_iou_pred_indices = -np.ones(len(objs_perspectives))
        for det_idx in range(len(objs_perspectives)):
            perspect_obj_iou_fmt = perspect_objs_iou_fmt[det_idx]

            ious_3d = evaluation.three_d_iou(perspect_obj_iou_fmt,
                                             ego_objs_iou_fmt)

            max_iou_3d = np.amax(ious_3d)
            max_ious_3d[det_idx] = max_iou_3d

            if max_iou_3d > cfg.IOU_MATCHING_THRESHOLD:
                max_iou_pred_indices[det_idx] = np.argmax(ious_3d)

        return max_ious_3d, max_iou_pred_indices

# Returns a list of lists of objects which have been matched
# Separates object types so matching doesn't attempt to match them together
def match_iou3ds(trust_objs, only_ego_matches):
    peds = []
    cars = []
    other = []

    for t_obj_list in trust_objs:
        curr_peds = []
        curr_cars = []
        curr_other = []
        for t_obj in t_obj_list:
            if t_obj.obj.type == 'Car':
                curr_cars.append(t_obj)
            elif t_obj.obj.type == 'Pedestrian':
                curr_peds.append(t_obj)
            else:
                # For now do not match other types as we aren't evaluating them
                #curr_other.append(t_obj)

        if len(curr_cars) > 0:
            cars.append(curr_cars)
        if len(curr_peds) > 0:
            peds.append(curr_peds)
        if len(curr_other) > 0:
            other.append(curr_other)

    matched = []
    if len(cars) > 0:
        matched_cars = match_obj_type(cars, only_ego_matches)
        for match_list in matched_cars:
            matched.append(match_list)
    if len(peds) > 0:
        matched_peds = match_obj_type(peds, only_ego_matches)
        for match_list in matched_peds:
            matched.append(match_list)
    if len(other) > 0:
        matched_other = match_obj_type(other, only_ego_matches)
        for match_list in matched_other:
            matched.append(match_list)

    return matched

# Call match_iou3ds to separate object types
# Returns a list of lists of objects which have been matched
def match_obj_type(trust_objs, only_ego_matches):
    matched_objs = []

    base_idx = 0

    #If only_ego_matches only try matching detections from the first trust_obj
    end_idx = len(trust_objs)
    if only_ego_matches:
        end_idx = min(end_idx, 1)

    for v_idx in range(0,end_idx):
        v_trust_objs = []

        # Add lists for all objs from v to matched_objs
        base_idx_increase = 0
        for trust_obj in trust_objs[v_idx]:
            if not trust_obj.matched:
                matched_objs.append([trust_obj])
                v_trust_objs.append(trust_obj)
                base_idx_increase += 1

        if base_idx_increase == 0:
            continue

        stripped_v_objs = trust_utils.strip_objs(v_trust_objs)

        for v2_idx in range(v_idx+1, len(trust_objs)):
            v2_trust_objs = trust_objs[v2_idx]
            stripped_v2_objs = trust_utils.strip_objs(v2_trust_objs)

            max_ious, iou_indices = get_iou3d_matches(stripped_v_objs, stripped_v2_objs)

            for obj_idx in range(0, len(iou_indices)):
                if iou_indices[obj_idx] != -1:
                    obj = v2_trust_objs[obj_idx]
                    if not obj.matched:
                        matched_idx = base_idx + int(iou_indices[obj_idx])
                        obj.matched = True
                        obj.matched_idx = matched_idx
                        matched_objs[matched_idx].append(obj)

        # Update index
        base_idx += base_idx_increase

    return matched_objs