import argparse
import numpy as np
import os
import sys

from wavedata.tools.obj_detection import obj_utils, evaluation

import config as cfg
import perspective_utils as p_utils

#Example usage:
#python filter_gt_labels.py --out_subdir='label_filtered_2'

def filter_gt_labels(out_subdir):
    """Plots detection errors for xyz, lwh, ry, and shows 3D IoU with
    ground truth boxes
    """

    classes = ['Car', 'Pedestrian']

    difficulty = 2
    out_dir = cfg.DATASET_DIR + '/' + out_subdir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Saving filtered labels to: ", out_dir)

    label_dir = cfg.DATASET_DIR + '/label_aug_2/'

    files = os.listdir(label_dir)
    num_files = len(files)
    sample_idx = 0
    for file in files:
        sys.stdout.flush()
        sys.stdout.write('\r{} / {}'.format(
            sample_idx + 1, num_files))
        sample_idx = sample_idx + 1

        filepath = label_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        # Get filtered ground truth
        all_gt_objs = obj_utils.read_labels(label_dir, idx, synthetic=True)
        if all_gt_objs != None:
            all_gt_objs = p_utils.filter_labels(
                                all_gt_objs,
                                max_dist=140)

        for obj in all_gt_objs:
            if obj.type == 'Person_sitting':
                obj.type = 'Pedestrian'
            print(obj.type)
            if not (obj.type == 'Pedestrian' or obj.type == 'Car'):
                obj.type = 'DontCare'
                print("Setting to DontCare")

        # Save gt to output file
        save_filtered_objs(all_gt_objs, idx, out_dir)

def save_filtered_objs(gt_objs, idx, out_dir):
    out_file = out_dir + '/{:06d}.txt'.format(idx)

    with open(out_file, 'w+') as f:
        if gt_objs is None:
            return
        for obj in gt_objs:
            occ_lvl = 0
            if obj.occlusion > 0.2:
                occ_lvl = 1
            if obj.occlusion > 0.5:
                occ_lvl = 2
            kitti_text_3d = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(obj.type,
                obj.truncation, occ_lvl, obj.alpha, obj.x1, obj.y1, obj.x2,
                obj.y2, obj.h, obj.w, obj.l, obj.t[0], obj.t[1], obj.t[2], obj.ry)

            f.write('%s\r\n' % kitti_text_3d)

def main():
    parser = argparse.ArgumentParser()

    # Example usage
    # --checkpoint_name='avod_exp_example'
    # --base_dir='/home/<username>/GTAData/'

    parser.add_argument('--out_subdir',
                        type=str,
                        dest='out_subdir',
                        required=True,
                        help='Output subdirectory must be specified.')


    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    filter_gt_labels(args.out_subdir)

main()