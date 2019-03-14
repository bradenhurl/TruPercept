import cv2
import numpy as np
import os
import sys
from wavedata.tools.obj_detection import obj_utils
import perspective_utils as p_utils
import matching_utils
import trust_utils

# Change this folder to point to the Trust Perception dataset
dataset_dir = os.path.expanduser('~') + '/wavedata-dev/demos/gta/training/'

def main():
    
    velo_dir = dataset_dir + 'velodyne'
    calib_dir = dataset_dir + 'calib'
    predictions_dir = dataset_dir + 'predictions'
    output_dir = dataset_dir + 'predictions_tru_percept'
    matching_dir = dataset_dir + 'matching_test'

    # Do this for every sample index
    velo_files = os.listdir(velo_dir)
    num_files = len(velo_files)
    file_idx = 0

    for file in velo_files:
        filepath = velo_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        # Load predictions from own vehicle
        #TODO Test if certainty values are corresponding correctly
        ego_preds = obj_utils.read_labels(predictions_dir, idx, results=True)
        ego_trust_objs = trust_utils.createTrustObjects(dataset_dir, idx, trust_utils.self_id, ego_preds)

        # Load predictions from nearby vehicles
        perspect_trust_objs = p_utils.get_all_detections(dataset_dir, idx, results=True)

        # Add fake detections to perspect_preds

        # Find matching pairs
        # Returns a list of lists of objects which have been matched
        perspect_trust_objs.insert(0, ego_trust_objs)
        matching_objs = matching_utils.match_iou3ds(perspect_trust_objs)

        # Print matching objects to test with visualization
        # out_file = matching_dir + '/{:06d}.txt'.format(idx)
        # if os.path.exists(out_file):
        #     os.remove(out_file)
        # else:
        #     print("Can not delete the file as it doesn't exists")
        # for match_list in matching_objs:
        #     if len(match_list) > 1:
        #         objs = trust_utils.strip_objs(match_list)
        #         save_filtered_objs(objs, idx, matching_dir)

        # Calculate trust from received detections

        # Adjust predictions

        # Add to vehicle trust

        sys.stdout.write("\rWorking on idx: {} / {}".format(
                file_idx + 1, num_files))
        sys.stdout.flush()
        file_idx = file_idx + 1



def save_filtered_objs(gt_objs, idx, out_dir):
    out_file = out_dir + '/{:06d}.txt'.format(idx)

    with open(out_file, 'a+') as f:
        if gt_objs is None:
            return
        for obj in gt_objs:
            kitti_text_3d = '{} {} {} {} {:d} {:d} {:d} {:d} {} {} {} {} {} {} {}'.format(obj.type,
                obj.truncation, obj.occlusion, obj.alpha, int(obj.x1), int(obj.y1), int(obj.x2),
                int(obj.y2), obj.h, obj.w, obj.l, obj.t[0], obj.t[1], obj.t[2], obj.ry)

            f.write('%s\n' % kitti_text_3d)


main()