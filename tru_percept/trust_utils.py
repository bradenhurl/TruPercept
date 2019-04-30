import os

from wavedata.tools.obj_detection import obj_utils

import certainty_utils
import config as cfg

self_id = 0

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
        self.evaluator_score = -1
        self.matched = False
        self.trust = 0.

class VehicleTrust:
    """The trust object unique per vehicle
    """

    def __init__(self, entity_id, trust_val, trust_messages):
        self.id = entity_id
        self.tVal = trust_val
        self.messages = trust_messages

# Returns the trust object of the vehicle from the own perspective (persp_dir)
def getPerspectiveOwnVehicleTrustObject(persp_dir, idx, persp_id):
    ego_dir = persp_dir + '/ego_object/'
    ego_detection = obj_utils.read_labels(ego_dir, idx)
    ego_detection[0].score = 1.0
    # TODO Filter object?
    ego_tDet = TrustDetection(persp_id, ego_detection[0], 1.0, 0)
    return ego_tDet


def createTrustObjects(persp_dir, idx, persp_id, detections, results):
    trust_detections = []

    # Add ego object (self)
    ego_tDet = getPerspectiveOwnVehicleTrustObject(persp_dir, idx, persp_id)
    trust_detections.append(ego_tDet)

    # Convert detections to trust objects
    if detections is not None and len(detections) > 0:
        certainties = certainty_utils.load_certainties(persp_dir, idx)

        c_idx = 0
        # Detection idx starts as 1 since own vehicle is detection with index 0
        det_idx = 1
        for det in detections:
            certainty = 1
            if results:
                pointsInBox = certainties[c_idx]
                certainty = certainty_utils.certainty_from_num_3d_points(pointsInBox)
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

# todo should we only update trust with messages we are certain of?
def get_message_trust_values(matching_objs, perspect_dir, perspect_id, idx):

    #TODO Should put a case if it is own vehicle but not any points in it
    for match_list in matching_objs:
        if len(matching_objs) > 1:
            for trust_obj in match_list:
                trust_obj.evaluator_id = perspect_id
                pointcloud = certainty_utils.get_nan_point_cloud(perspect_dir, idx)
                trust_obj.evaluator_certainty = certainty_utils.numPointsIn3DBox(trust_obj.obj, pointcloud, perspect_dir, idx)
                