import os
import sys

import config as cfg
import constants as const
# import false_detections as false_dets
import perspective_utils as p_utils
import std_utils
import trust_utils

def save_false_dets():

    print("Beginning save of false detections")

    # First for the ego vehicle
    save_false_dets_persp(cfg.DATASET_DIR, const.ego_id())

    # Then for all the alternate perspectives
    persp_count = len(os.listdir(cfg.ALT_PERSP_DIR))
    persp_idx = 0
    for entity_str in const.valid_perspectives():
        persp_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)
        save_false_dets_persp(persp_dir, int(entity_str))

        sys.stdout.flush()
        sys.stdout.write('\rFinished saving detections for perspective {}: {} / {}'.format(
            int(entity_str), persp_idx, persp_count))
        persp_idx += 1


def save_false_dets_persp(persp_dir, persp_id):
    velo_dir = persp_dir + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    out_dir = persp_dir + cfg.FALSE_DETECTIONS_SUBSUBDIR
    std_utils.make_dir(out_dir)

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        # Load predictions (and potentially false detections) from own vehicle
        trust_objs = p_utils.get_detections(persp_dir, persp_dir, idx, \
                                        persp_id, persp_id, results=True, filter_area=False,
                                        override_load=True)

        objs = trust_utils.strip_objs(trust_objs)
        # Save the detections (which were randomly modified)
        # so the same random modifications can be used by others
        std_utils.save_objs_to_file(objs, idx, out_dir, results=True)

save_false_dets()