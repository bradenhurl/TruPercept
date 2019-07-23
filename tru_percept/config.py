import os
import sys
import logging

# Base dataset directory
SCENE_NUM = 5
DATASET_DIR = os.path.expanduser('~') + '/GTAData/TruPercept/object_tru_percept{}/training'.format(SCENE_NUM)


# ********************************************************************** #
# Experimentation configuration

# Certainty/visibility threshold scores
# TODO Probabilistic approach to certainty
gamma_upper = 100
gamma_lower = 0
gamma_upper_peds = 40
gamma_lower_peds = 0

# Filters final detections by this score threshold to speed up kitti evaluation
SCORE_THRESHOLD = 0.09

# Minimum iou to be considered a match
IOU_MATCHING_THRESHOLD = 0.1

# False detections to use
# None for no detections
# malicious_front for adding detections in front of vehicles
FALSE_DETECTIONS_TYPE = 'malicious_front'
FALSE_DETECTIONS_SUBDIR = 'false_detections'

# Default trust value for first-time vehicles
DEFAULT_VEHICLE_TRUST_VAL = 0.5

# Message evaluation value for negative matches
NEG_EVAL_SCORE = 0.0

# Use regular vs filtered (for distance, etc) ground truth labels
EVALUATE_UNFILTERED = True

# Aggregation method
# 0 is averaging
# 1 is additive
# 2 is based on msg evaluations
# 3 is believe all with score 1.0
# 4 is believe all with score 1.0 unless non-matching, then confidence
# 5 is believe all ego which are matches
# 6 is believe all ego which have matches at score = 1, believe other ego
#           with a score they were detected with
# 7 is sanity check (passes through ego vehicle objects with same score)
# 9 Same as 2 but average msg eval with ego vehicle detection confidence
AGGREGATE_METHOD = 9

# Attempts to synchronize the detections by matching and using velocity
SYNCHRONIZE_DETS = True

# If True, excludes perspectives which are not type 'Car'
EXCLUDE_OTHER_VEHICLE_TYPES = True

# If True, aggregates message evals with own detection
INCLUDE_OWN_DETECTION_IN_EVAL = True

# ********************************************************************** #
# For testing

# Min/max indices which will be run (for running tests primarily)
# If set to 0 and sys.maxsize, respectively, all indices will be run
# when present, up to the maximum index in the DATASET_DIR/velodyne folder
MIN_IDX = 0
MAX_IDX = sys.maxsize
# MIN_IDX = 7
#MAX_IDX = 14

# Test index can override min and max for testing single frame
TEST_IDX = -1#58#-1#14#59
if TEST_IDX != -1:
    MIN_IDX = TEST_IDX
    MAX_IDX = TEST_IDX

# Set to false to only load ground truth instead of predictions
USE_RESULTS = True

# Set to true to visualize points_in_3d_boxes during calculation
VISUALIZE_POINTS_IN_3D_BOXES = False

# Set to true to visualize matches during message evaluations
VISUALIZE_MATCHES = False

# Set to true to visualize message evaluation scores from each perspective
VISUALIZE_MSG_EVALS = False

# Set to true to visualize aggregated msg evaluation scores from the ego vehicle perspective
VISUALIZE_AGG_EVALS = False

# Set to true to visualize the final detections with scores
VISUALIZE_FINAL_DETS = False

# Set and initialize logging
LOG_LVL = logging.DEBUG
LOG_FILE = DATASET_DIR + '/log.txt'
# Initialize logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LVL, format='%(levelname)s: %(filename)s(%(lineno)d): %(message)s')


# ********************************************************************** #
# Don't need to change these ones

# Subdirectories for storing intermediate steps
POINTS_IN_3D_BOXES_DIR = 'points_in_3d_boxes'
MSG_EVALS_SUBDIR = 'msg_evals'
AGG_MSG_EVALS_SUBDIR = 'agg_msg_evals' # Aggregated msg evals
V_TRUST_SUBDIR = 'vehicle_trust_scores'
FINAL_DETS_SUBDIR = 'final_detections'

if not USE_RESULTS:
    POINTS_IN_3D_BOXES_DIR += '_gt'
    MSG_EVALS_SUBDIR += '_gt'
    V_TRUST_SUBDIR += '_gt'
    FINAL_DETS_SUBDIR += '_gt'
    AGG_MSG_EVALS_SUBDIR += '_gt'

# Area filtered subdir for kitti evaluation
FINAL_DETS_SUBDIR_AF = 'final_detections_area_filtered'

SYNCHRONIZED_PREDS_DIR = 'predictions_synchro'

KITTI_EVAL_SUBDIR = 'kitti_native_eval'

AVOD_OUTPUT_DIR = 'predictions'
PREDICTIONS_SUBDIR = 'predictions'

# Use regular vs filtered (for distance, etc) ground truth labels
if EVALUATE_UNFILTERED:
    LABEL_DIR = 'label_2'
else:
    LABEL_DIR = 'label_filtered_2'

ALT_PERSP_DIR = DATASET_DIR + '/alt_perspective/'


# ********************************************************************** #
# Set once

# Set this to proper device then leave it
CUDA_DEVICE = '0'

# Only skips indices for inference!!!
# Indices to skip in case of bugs, problems
INDICES_TO_SKIP = {}
if SCENE_NUM == 3:
    INDICES_TO_SKIP = {267,268,269,270}#Bugged for pedestrian detection
    #INDICES_TO_SKIP += {12,13,14}#Detections not properly synchronized