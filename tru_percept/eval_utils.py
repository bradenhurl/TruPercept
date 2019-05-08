import os
import subprocess

import config as cfg
import constants as const

def run_kitti_native_script(score_threshold):
    """Runs the kitti native code script."""

    script_folder = const.top_dir() + \
        '/kitti_native_eval/'
    run_script = script_folder + 'run_eval.sh'
    script_file = script_folder + 'evaluate_object_3d_offline'

    # Checks if c++ code is compiled
    if not os.path.isfile(script_file):
        # run the script to compile the c++ code
        make_script = script_folder + 'run_make.sh'
        subprocess.call([make_script, script_folder])

    # Sets up the eval to output in the dataset directory
    predictions_dir = cfg.DATASET_DIR + '/' + cfg.FINAL_DETS_SUBDIR
    label_dir = cfg.DATASET_DIR + '/' + cfg.LABEL_DIR

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    subprocess.call([run_script, script_folder,
                     str(score_threshold),
                     str(predictions_dir),
                     str(cfg.FINAL_DETS_SUBDIR),
                     str(cfg.DATASET_DIR),
                     str(label_dir)])