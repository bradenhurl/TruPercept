import os
import numpy as np

import config as cfg
import constants as const
import perspective_utils as p_utils
import plausibility_checker
import points_in_3d_boxes as points_3d
from tools.visualization import vis_objects

def main():
    velo_dir = cfg.DATASET_DIR + '/velodyne'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
            continue

        ego_dir = p_utils.get_folder(const.ego_id())
        gt_objects = p_utils.get_detections(ego_dir, ego_dir, idx, const.ego_id(), const.ego_id(), results=False, filter_area=True)

        pc = points_3d.get_nan_point_cloud(ego_dir, idx)

        obj_invalid = []
        obj_valid = []

        for obj in gt_objects:
            if not (obj.obj.type == 'Car' or obj.obj.type == 'Pedestrian'):
                continue

            obj_pos = np.asanyarray(obj.obj.t)
            obj_dist = np.sqrt(np.dot(obj_pos, obj_pos.T))
            num_points = points_3d.numPointsIn3DBox(obj.obj, pc, ego_dir, idx)
            if plausibility_checker.is_plausible(obj.obj, idx, const.ego_id(), obj.det_idx) == False:
                print("Not plausible at dist: ", obj_dist, num_points)
                obj_invalid.append(obj.obj)
            else:
                # print("Plausible at dist: ", obj_dist, num_points)
                obj_valid.append(obj.obj)

        if len(obj_invalid) > 0:
            vis_objects.visualize_objects(obj_invalid, idx, False, False, -1, compare_with_gt=False, show_image=False)


main()