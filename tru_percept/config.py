import os
import sys
import logging

# Base dataset directory
DATASET_DIR = os.path.expanduser('~') + '/GTAData/TruPercept/object_tru_percept3/training'

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
#MAX_IDX = 6

# Only skips indices for inference!!!
# Indices to skip in case of bugs, problems
INDICES_TO_SKIP = {267,268,269,270}

SCORE_THRESHOLD = 0.1

# Default trust value for first-time vehicles
DEFAULT_VEHICLE_TRUST_VAL = 0.5

CUDA_DEVICE = '0'

USE_RESULTS = True

# Aggregation method
# 0 is averaging
# 1 is additive
AGGREGATE_METHOD = 2

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
LOG_LVL = logging.INFO
LOG_FILE = DATASET_DIR + '/log.txt'
# Initialize logging
logging.basicConfig(filename=LOG_FILE, level=LOG_LVL, format='%(levelname)s: %(filename)s(%(lineno)d): %(message)s')