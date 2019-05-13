import os
import subprocess
from distutils.dir_util import copy_tree

import config as cfg
import constants as const
import std_utils

def run_kitti_native_script(score_threshold, evaluate_avod=True):
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
    avod_output_dir = cfg.DATASET_DIR + '/' + cfg.AVOD_OUTPUT_DIR

    # Round this because protobuf encodes default values as full decimal
    score_threshold = round(score_threshold, 3)

    # copy predictions into proper kitti format
    preds_eval_dir = cfg.DATASET_DIR + '/' + cfg.KITTI_EVAL_SUBDIR + '/' + cfg.FINAL_DETS_SUBDIR
    avod_eval_dir = cfg.DATASET_DIR + '/' + cfg.KITTI_EVAL_SUBDIR + '/' + cfg.AVOD_OUTPUT_DIR

    std_utils.make_dir(preds_eval_dir)
    copy_tree(predictions_dir, preds_eval_dir + '/data')

    print("*********************************************************************\n" +
          "Results from tru_percept: \n" +
          "*********************************************************************\n")
    subprocess.call([run_script, script_folder,
                     str(score_threshold),
                     str(preds_eval_dir),
                     str(cfg.FINAL_DETS_SUBDIR),
                     str(cfg.DATASET_DIR),
                     str(label_dir)])

    if evaluate_avod:
        std_utils.make_dir(avod_eval_dir)
        copy_tree(avod_output_dir, avod_eval_dir + '/data')
        print("\n\n\n*********************************************************************\n" +
              "Results from AVOD: \n" +
              "*********************************************************************\n")
        subprocess.call([run_script, script_folder,
                         str(score_threshold),
                         str(avod_eval_dir),
                         str(cfg.FINAL_DETS_SUBDIR),
                         str(cfg.DATASET_DIR),
                         str(label_dir)])