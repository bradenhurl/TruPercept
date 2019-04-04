from wavedata.tools.obj_detection import obj_utils
import preprocessing.certainty_utils as certainty_utils

self_id = 0
DEFAULT_TRUST_VAL = 0.5

# Dictionary for vehicle trust values
trust_map = {}


class TrustDetection:
    """A single detection 
    ObjectLabel from obj_utils
    3D Points in Box (integer value)
    Certainty - 
    Matched (Boolean) set to true if it was matched
    """

    def __init__(self, entity_id, obj, points):
        self.id = entity_id
        self.obj = obj
        self.pointsInBox = points
        self.matched = False
        self.trust = 0.

class VehicleTrust:
    """The trust object unique per vehicle
    """

    def __init__(self, entity_id, trust_val, trust_messages):
        self.id = entity_id
        self.tVal = trust_val
        self.messages = trust_messages

def getEgoTrustObject(base_dir, idx, entity_id):
    ego_dir = base_dir + '/ego_object/'
    ego_detection = obj_utils.read_labels(ego_dir, idx)
    ego_detection[0].score = 1.0
    # TODO Filter object?
    ego_tDet = TrustDetection(entity_id, ego_detection[0], 1.0)
    return ego_tDet


def createTrustObjects(base_dir, idx, entity_id, detections):
    trust_detections = []

    # Add ego object (self)
    ego_tDet = getEgoTrustObject(base_dir, idx, entity_id)
    trust_detections.append(ego_tDet)

    # Convert detections to trust objects
    if detections is not None and len(detections) > 0:
        certainties = certainty_utils.load_certainties(base_dir, idx)

        c_idx = 0
        for det in detections:
            tDet = TrustDetection(entity_id, det, certainties[c_idx])
            trust_detections.append(tDet)
            c_idx += 1

    return trust_detections

def strip_objs(trust_objs):
    stripped_objs = []
    if trust_objs is not None:
        for trust_obj in trust_objs:
            stripped_objs.append(trust_obj.obj)

    return stripped_objs

# todo should we only update trust with messages we are certain of?
def get_message_trust_values(matching_objs):

    for match_list in matching_objs:
        if len(matching_objs) > 1:
            v_id = matching_objs[0].id
            if v_id in trust_map:
                trust = trust_map[v_id].trust_val
            else:
                trust_map[v_id] = DEFAULT_TRUST_VAL

        for trust_obj in matching_objs:
            # TODO Need to update based on 


def trust_from_message(trust_msg, related_trust_msgs):

