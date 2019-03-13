from wavedata.tools.obj_detection import obj_utils
import certainty_utils

self_id = 0

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

def getEgoTrustObject(base_dir, idx, entity_id):
    ego_dir = base_dir + '/ego_object/'
    ego_detection = obj_utils.read_labels(ego_dir, idx)
    ego_detection[0].score = 1.0
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