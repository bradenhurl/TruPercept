from wavedata.tools.obj_detection import obj_utils
import preprocessing.certainty_utils as certainty_utils
import os
import shutil
import numpy as np
import perspective_utils

self_id = 0
DEFAULT_TRUST_VAL = 0.5
MSG_EVALS_SUBDIR = 'msg_evals'

# Dictionary for vehicle trust values
trust_map = {}


class TrustDetection:
    """A single detection 
    ObjectLabel from obj_utils
    3D Points in Box (integer value)
    Matched (Boolean) set to true if it was matched
    Evaluated trust from the message
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


def createTrustObjects(base_dir, idx, entity_id, detections, results):
    trust_detections = []

    # Add ego object (self)
    ego_tDet = getEgoTrustObject(base_dir, idx, entity_id)
    trust_detections.append(ego_tDet)

    # Convert detections to trust objects
    if detections is not None and len(detections) > 0:
        certainties = certainty_utils.load_certainties(base_dir, idx)

        c_idx = 0
        for det in detections:
            certainty = 1
            if results:
                certainty = certainties[c_idx]
            tDet = TrustDetection(entity_id, det, certainty)
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
def get_message_trust_values(matching_objs, perspect_dir, idx):

    #TODO finish this
    for match_list in matching_objs:
        if len(matching_objs) > 1:
            for trust_obj in match_list:
                pointcloud = certainty_utils.get_nan_point_cloud(perspect_dir, idx)
                trust_obj.pointsInBox = certainty_utils.numPointsIn3DBox(trust_obj.obj, pointcloud, perspect_dir, idx)
                


def get_vehicle_trust_value(entity_id):
    if v_id in trust_map:
        trust = trust_map[v_id].trust_val
    else:
        trust_map[v_id] = DEFAULT_TRUST_VAL

def trust_from_message(trust_msg, related_trust_msgs):
    #TODO
    print("Trust from msg")

def save_msg_evals(msg_trusts, base_dir, ego_id, idx):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Save msg evals in trust")
    if msg_trusts is None:
        print("Msg trusts is none")
        return

    for matched_msgs in msg_trusts:
        first_obj = True
        print("Outputting list of matched objects")
        for trust_obj in matched_msgs:
            if first_obj:
                #Skip first object as it is from self
                first_obj = False
                print("Skipping first object")
                continue

            # Fill the array to write
            msg_trust_output = np.zeros([1, 4])
            #TODO - fill in correct values
            msg_trust_output[0,0] = 0#trust_obj.msg_id
            msg_trust_output[0,1] = 1#trust_obj.confidence
            msg_trust_output[0,2] = trust_obj.pointsInBox#certainty
            msg_trust_output[0,3] = ego_id

            print("********************Saving trust val to id: ", trust_obj.id, " at idx: ", idx)
            # Save to text file
            file_path = perspective_utils.get_folder(base_dir, ego_id, trust_obj.id) + '/{}/{:06d}.txt'.format(MSG_EVALS_SUBDIR,idx)
            print("Writing msg evals to file: ", file_path)
            make_dir(file_path)
            with open(file_path, 'a+') as f:
                np.savetxt(f, msg_trust_output,
                           newline='\r\n', fmt='%s')

def make_dir(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def delete_msg_evals(base_dir):
    dirpath = os.path.join(base_dir, MSG_EVALS_SUBDIR)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    altPerspect_dir = base_dir + '/alt_perspective/'
    for entity_str in os.listdir(altPerspect_dir):
        perspect_dir = os.path.join(altPerspect_dir, entity_str)
        dirpath = os.path.join(perspect_dir, MSG_EVALS_SUBDIR)
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            print("Deleting directory: ", dirpath)
            shutil.rmtree(dirpath)