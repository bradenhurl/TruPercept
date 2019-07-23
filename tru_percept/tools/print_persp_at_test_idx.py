import os

import config as cfg
import constants as const

# For all the alternate perspectives
for entity_str in const.valid_perspectives():
    perspect_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)

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