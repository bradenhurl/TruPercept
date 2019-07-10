import os
import sys
import logging

# Base dataset directory
SCENE_NUM = 3
DATASET_DIR = os.path.expanduser('~') + '/GTAData/TruPercept/object_tru_percept{}/training'.format(SCENE_NUM)

# Certainty threshold scores
# TODO Probabilistic approach to certainty
gamma_upper = 500
gamma_lower = 10

# Min/max indices which will be run (for running tests primarily)
# If set to 0 and sys.maxsize, respectively, all indices will be run
# when present, up to the maximum index in the DATASET_DIR/velodyne folder
MIN_IDX = 0
MAX_IDX = sys.maxsize
# MIN_IDX = 7
#MAX_IDX = 14

# Test index can override min and max for testing single frame
TEST_IDX = -1#13#59
if TEST_IDX != -1:
    MIN_IDX = TEST_IDX
    MAX_IDX = TEST_IDX

# Only skips indices for inference!!!
# Indices to skip in case of bugs, problems
INDICES_TO_SKIP = {}
if SCENE_NUM == 3:
    INDICES_TO_SKIP = {267,268,269,270}#Bugged for pedestrian detection
    #INDICES_TO_SKIP += {12,13,14}#Detections not properly synchronized

SCORE_THRESHOLD = 0.1

IOU_MATCHING_THRESHOLD = 1.0

# Default trust value for first-time vehicles
DEFAULT_VEHICLE_TRUST_VAL = 0.5

# Message evaluation value for negative matches
NEG_EVAL_SCORE = 0.0

CUDA_DEVICE = '0'

USE_RESULTS = True

# Aggregation method
# 0 is averaging
# 1 is additive
# 2 is based on msg evaluations
# 3 is believe all with score 1.0
# 4 is believe all with score 1.0 unless non-matching, then confidence
# 5 is believe all ego which are matches
AGGREGATE_METHOD = 7

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

# Use regular vs filtered (for distance, etc) ground truth labels
LABEL_DIR = 'label_filtered_2'
#LABEL_DIR = 'label_2'

AVOD_OUTPUT_DIR = 'predictions'
KITTI_EVAL_SUBDIR = 'kitti_native_eval'

# Set and initialize logging
LOG_LVL = logging.DEBUG
LOG_FILE = DATASET_DIR + '/log.txt'
# Initialize logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LVL, format='%(levelname)s: %(filename)s(%(lineno)d): %(message)s')