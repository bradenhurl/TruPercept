import os
import sys

# Base dataset directory
DATASET_DIR = os.path.expanduser('~') + '/wavedata-dev/demos/gta/training/'

# Certainty threshold scores
# TODO Probabilistic approach to certainty
gamma_upper = 500
gamma_lower = 10

# Min/max indices which will be run (for running tests primarily)
# If set to 0 and sys.maxsize, respectively, all indices will be run
# when present, up to the maximum index in the DATASET_DIR/velodyne folder
#MIN_IDX = 0
#MAX_IDX = sys.maxsize
MIN_IDX = 7
MAX_IDX = 10

SCORE_THRESHOLD = 0.01

# Default trust value for first-time vehicles
DEFAULT_VEHICLE_TRUST_VAL = 0.5

# Subdirectories for storing intermediate steps
MSG_EVALS_SUBDIR = 'msg_evals'
V_TRUST_SUBDIR = 'vehicle_trust_scores'
FINAL_DETS_SUBDIR = 'final_detections'