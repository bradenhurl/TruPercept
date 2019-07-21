import os
import math

from wavedata.tools.obj_detection import obj_utils

import certainty_utils
import config as cfg
import constants as const
import points_in_3d_boxes as points_3d
import perspective_utils as p_utils

# Dictionary for vehicle trust values
trust_map = {}


class TrustDetection:
    """A single detection 
    obj                 ObjectLabel from obj_utils
    det_idx             Index of detections (Starting from 0 is self)
    detector_id         entity_id of vehicle which perceived the detection
    evaluator_id        entity_id of vehicle which is evaluating (received) the detection
    evaluator_certainty certainty_score of evaluation
    evaluator_score     Unused right now, should be used to signify if evaluator 
                        believes detection is true or not
    """

    def __init__(self, persp_id, obj, persp_certainty, det_idx):
        self.obj = obj
        self.det_idx = det_idx
        self.detector_id = persp_id
        self.detector_certainty = persp_certainty
        self.evaluator_id = -1
        self.evaluator_certainty = -1
        self.evaluator_3d_points = -1
        self.evaluator_score = -1
        self.matched = False
        self.trust = 0.

class MessageEvaluation:
    """Everything from an evaluated message
    det_idx                 Index of the detection
    evaluator_id            entity_id of the evaluator vehicle
    evaluator_certainty     certainty
    evaluator_certainty certainty_score of evaluation
    evaluator_score     Unused right now, should be used to signify if evaluator 
                        believes detection is true or not
    """

    def __init__(self):
        self.det_idx = -1
        self.detector_score = -1
        self.detector_certainty = -1
        self.evaluator_id = -1
        self.evaluator_certainty = -1
        self.evaluator_score = -1

class AggregatedMessageEvaluation:
    """Everything from an aggrgated message evaluation
    detector_id             entity_id of vehicle which perceived the detection
    det_idx                 Index of the detection
    msg_trust               Trust of msg evaluated from all vehicles
    """

    def __init__(self):
        self.detector_id = -1
        self.det_idx = -1
        self.msg_trust = -1


class VehicleTrust:
    """Trust for a vehicle
    val                 Trust value
    sum                 Sum of numerator
    count               Count of evaluations
    """

    def __init__(self):
        self.val = cfg.DEFAULT_VEHICLE_TRUST_VAL
        self.sum = 0.
        self.count = 0.


def createTrustObjects(persp_dir, idx, persp_id, detections, results, to_persp_dir):
    trust_detections = []

    #points_dict = points_3d.load_points_in_3d_boxes(idx, persp_id)
    # TODO - set pointsInBox for evaluator and detector

    # Convert detections to trust objects
    if detections is not None and len(detections) > 0:
        # TODO - load_certainties once cerainty calculation is more complex
        #certainties = certainty_utils.load_certainties(persp_dir, idx)

        c_idx = 0
        # Detection idx starts as 1 since own vehicle is detection with index 0
        det_idx = 1
        first_det = True
        for det in detections:
            if first_det:
                # Add ego object (self) - it is always first in detections list
                ego_tDet = TrustDetection(persp_id, det, 1.0, 0)
                trust_detections.append(ego_tDet)
                first_det = False
            else:
                certainty = 1
                if results:
                    pointsInBox = -1#points_dict[persp_id, det_idx]
                    certainty = -1#certainty_utils.certainty_from_3d_points(pointsInBox)
                tDet = TrustDetection(persp_id, det, certainty, det_idx)
                trust_detections.append(tDet)
                c_idx += 1
                det_idx += 1

    return trust_detections

def strip_objs(trust_objs):
    stripped_objs = []
    if trust_objs is not None:
        for trust_obj in trust_objs:
            stripped_objs.append(trust_obj.obj)

    return stripped_objs

def strip_objs_lists(trust_objs_lists):
    stripped_objs = []
    for obj_list in trust_objs_lists:
        stripped_list = strip_objs(obj_list)
        for stripped_obj in stripped_list:
            stripped_objs.append(stripped_obj)
    return stripped_objs

# todo should we only update trust with messages we are certain of?
def get_message_trust_values(matching_objs, persp_dir, persp_id, idx):
    points_dict = points_3d.load_points_in_3d_boxes(idx, persp_id)

    #TODO Should put a case if it is own vehicle but not any points in it
    for match_list in matching_objs:
        # TODO - Need to find a way to put in negative message evaluations
        # Likely add a score of 0 for non-matching detections
        # Need to properly set certainty for these detections
        # Distance and pointsIn3DBox based
        if len(match_list) >= 1:
            for trust_obj in match_list:
                trust_obj.evaluator_id = persp_id
                trust_obj.evaluator_3d_points = points_dict[trust_obj.detector_id, trust_obj.det_idx]
                trust_obj.evaluator_certainty = certainty_utils.certainty_from_3d_points(trust_obj.evaluator_3d_points)
                # TODO Also try incorporating a matching score? Based on IoU
                # Set the score (confidence) to be the same as the matched detection
                if match_list[0].detector_id == persp_id:
                    trust_obj.evaluator_score = match_list[0].obj.score
                else:
                    trust_obj.evaluator_score = cfg.NEG_EVAL_SCORE
                