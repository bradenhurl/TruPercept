import os
import errno
import shutil
import logging

import config as cfg

def make_dir(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def delete_all_subdirs(subdir):
    delete_subdir(subdir)

    altPerspect_dir = cfg.DATASET_DIR + '/alt_perspective/'
    for entity_str in os.listdir(altPerspect_dir):
        perspect_dir = os.path.join(altPerspect_dir, entity_str)
        delete_subdir(subdir, perspect_dir)

def delete_subdir(subdir, basedir=cfg.DATASET_DIR):
    dirpath = os.path.join(basedir, subdir)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        logging.debug("Deleting directory: ", dirpath)
        shutil.rmtree(dirpath)
