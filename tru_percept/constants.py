import os
import config as cfg
from wavedata.tools.obj_detection import obj_utils
import logging


# Constants for x, y, z (standard x,y,z notation)
X = [1., 0., 0.]
Y = [0., 1., 0.]
Z = [0., 0., 1.]

ALT_PERSP_DIR = cfg.DATASET_DIR + '/alt_perspective/'

_ego_id = int(-1)

def ego_id():
    global _ego_id

    # Obtain the ego ID if not set yet
    if _ego_id == -1:
        ego_folder = cfg.DATASET_DIR + '/ego_object'
        ego_info = obj_utils.read_labels(ego_folder, 0, synthetic=True)
        _ego_id = int(ego_info[0].id)
        logging.info("Setting ego ID as: {:d}".format(_ego_id))

    return _ego_id


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    tru_percept_root_dir = root_dir()
    return os.path.split(tru_percept_root_dir)[0]

# Returns a list of all valid perspective entity strings
def valid_perspectives():
    entity_strings = []
    for entity_str in os.listdir(ALT_PERSP_DIR):
        persp_dir = os.path.join(ALT_PERSP_DIR, entity_str)
        if not os.path.isdir(persp_dir):
            continue

        # Check if it is valid type (only cars are valid)
        if cfg.EXCLUDE_OTHER_VEHICLE_TYPES:
            ego_dir = persp_dir + '/ego_object/'
            indices = os.listdir(ego_dir)
            idx = int(indices[0].split('.')[0])
            ego_detection = obj_utils.read_labels(ego_dir, idx)
            if ego_detection[0].type != 'Car':
                continue

        entity_strings.append(entity_str)

    return entity_strings