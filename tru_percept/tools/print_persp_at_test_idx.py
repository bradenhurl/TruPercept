import os

import config as cfg

# Then for all the alternate perspectives
alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'

for entity_str in os.listdir(alt_pers_dir):
    perspect_dir = os.path.join(alt_pers_dir, entity_str)
    if not os.path.isdir(perspect_dir):
        continue

    velo_dir = perspect_dir + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        print(entity_str)
        break