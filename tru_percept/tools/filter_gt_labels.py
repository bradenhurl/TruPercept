import os
import sys
import argparse

import numpy as np
from wavedata.tools.obj_detection import obj_utils, evaluation

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_encoder

#Example usage:
#python filter_gt_labels.py --checkpoint_name='pyramid_cars_gta' --base_dir='/home/<username>/GTAData/'

def filter_gt_labels(dataset_config, base_dir):
    """Plots detection errors for xyz, lwh, ry, and shows 3D IoU with
    ground truth boxes
    """

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_dir = base_dir
    dataset_config.dataset_dir = base_dir

    dataset_config.data_split = 'train'
    dataset_config.data_split_dir = 'training'

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    dataset.classes = ['Car', 'Bus', 'Pedestrian', 'Cyclist', 'Truck']
    dataset.aug_sample_list = []

    difficulty = 2
    out_dir = base_dir + dataset_config.data_split_dir + '/label_filtered_2/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Saving filtered labels to: ", out_dir)
    
    # TODO read in augmented labels to filter by 3D point count
    # Also changes synthetic to true in read_labels
    # dataset.label_dir = base_dir + '/' + dataset_config.data_split_dir + '/label_aug_2/'
    dataset.label_dir = base_dir + '/' + dataset_config.data_split_dir + '/label_2/'

    files = os.listdir(dataset.label_dir)
    num_files = len(files)
    sample_idx = 0
    for file in files:
        sys.stdout.flush()
        sys.stdout.write('\r{} / {}'.format(
            sample_idx + 1, num_files))
        sample_idx = sample_idx + 1

        filepath = dataset.label_dir + '/' + file
        idx = int(os.path.splitext(file)[0])

        # Get filtered ground truth
        all_gt_objs = obj_utils.read_labels(dataset.label_dir, idx, synthetic=False)
        if all_gt_objs != None:
            all_gt_objs = dataset.kitti_utils.filter_labels(
                all_gt_objs, max_forward=dataset.kitti_utils.area_extents[2,1],
                max_side=dataset.kitti_utils.area_extents[0,1])

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

    parser.add_argument('--checkpoint_name',
                        type=str,
                        dest='checkpoint_name',
                        required=True,
                        help='Checkpoint name must be specified as a str\
                        and must match the experiment config file name.')

    parser.add_argument('--base_dir',
                        type=str,
                        dest='base_dir',
                        required=True,
                        help='Base data directory must be specified')


    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    experiment_config = args.checkpoint_name + '.config'

    # Read the config from the experiment folder
    experiment_config_path = avod.root_dir() + '/data/outputs/' +\
        args.checkpoint_name + '/' + experiment_config

    model_config, _, eval_config, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            experiment_config_path, is_training=False)

    filter_gt_labels(dataset_config, args.base_dir)

main()