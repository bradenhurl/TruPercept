"""Detection model inferencing.

This runs the DetectionModel evaluator in test mode to output detections.
"""

import argparse
import os
import sys

import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core.evaluator import Evaluator

import estimate_ground_planes
import create_split
import save_kitti_predictions
import certainty_utils

#python run_inference_alt_perspectives.py --checkpoint_name='pyramid_people_gta_40k_constantlr' --ckpt_indices=98 --base_dir='/home/bradenhurl/wavedata-dev/demos/gta/'


def inference(model_config, eval_config,
              dataset_config, base_dir,
              ckpt_indices, additional_cls):

    print("Additional class: ", additional_cls)
    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    dataset_config.data_dir = base_dir
    dataset_config.dataset_dir = base_dir

    dataset_config.data_split = 'train'
    dataset_config.data_split_dir = 'training'

    eval_config.eval_mode = 'test'
    eval_config.evaluate_repeatedly = False

    dataset_config.has_labels = False
    # Enable this to see the actually memory being used
    eval_config.allow_gpu_mem_growth = True

    eval_config = config_builder.proto_to_obj(eval_config)
    # Grab the checkpoint indices to evaluate
    eval_config.ckpt_indices = ckpt_indices

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_list = []

    # Setup the model
    # Overwrite repeated field
    model_config = config_builder.proto_to_obj(model_config)
    # Switch path drop off during evaluation
    model_config.path_drop_probabilities = [1.0, 1.0]

    # Create validation split
    create_split.create_split(dataset_config.dataset_dir, dataset_config.dataset_dir + dataset_config.data_split_dir + '/', 'val')

    inferPerspective(model_config, eval_config, dataset_config, additional_cls)

    altPerspect_dir = base_dir + dataset_config.data_split_dir + '/alt_perspective/'

    p_idx = 0
    p_count = len(os.listdir(altPerspect_dir))
    for entity_str in os.listdir(altPerspect_dir):
        if not os.path.isdir(os.path.join(altPerspect_dir, entity_str)):
            continue

        dataset_config.data_split = entity_str
        dataset_config.data_split_dir = entity_str
        dataset_config.dataset_dir = altPerspect_dir
        inferPerspective(model_config, eval_config, dataset_config, additional_cls)
        p_idx += 1
        print('\n\n********************Finished perspective: {} / {} ***********************\n\n'.format(
            p_idx, p_count))

def inferPerspective(model_config, eval_config, dataset_config, additional_cls):
    model_name = model_config.model_name

    entity_perspect_dir = dataset_config.dataset_dir + dataset_config.data_split_dir + '/'

    print("Inferring perspective: ", dataset_config.data_split, entity_perspect_dir, dataset_config.dataset_dir)

    files_in_range = create_split.create_split(dataset_config.dataset_dir, entity_perspect_dir, dataset_config.data_split)
    
    # If there are no files within the range cfg.MIN_IDX, cfg.MAX_IDX
    # then skip this perspective
    if not files_in_range:
        print("No files within the range cfg.MIN_IDX, cfg.MAX_IDX, skipping perspective")
        return

    if not additional_cls:
        estimate_ground_planes.estimate_ground_planes(entity_perspect_dir, dataset_config, 0)

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    #Switch inference output directory
    model_config.paths_config.pred_dir = entity_perspect_dir + '/predictions/'
    print("Prediction directory: ", model_config.paths_config.pred_dir)

    with tf.Graph().as_default():
        if model_name == 'avod_model':
            model = AvodModel(model_config,
                              train_val_test=eval_config.eval_mode,
                              dataset=dataset)
        elif model_name == 'rpn_model':
            model = RpnModel(model_config,
                             train_val_test=eval_config.eval_mode,
                             dataset=dataset)
        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        model_evaluator = Evaluator(model,
                                    dataset_config,
                                    eval_config)
        model_evaluator.run_latest_checkpoints()

    save_kitti_predictions.convertPredictionsToKitti(dataset, model_config.paths_config.pred_dir, additional_cls)
    certainty_utils.save_num_points_in_3d_boxes(entity_perspect_dir, additional_cls)

def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    # --checkpoint_name='avod_exp_example'
    # --data_split='test'
    # --ckpt_indices=50 100 112
    # Optional arg:
    # --device=0
    # --additional_cls=False

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

    parser.add_argument(
        '--ckpt_indices',
        type=int,
        nargs='+',
        dest='ckpt_indices',
        required=True,
        help='Checkpoint indices must be a set of \
        integers with space in between -> 0 10 20 etc')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='0',
                        help='CUDA device id')

    parser.add_argument('--additional_cls',
                        dest='additional_cls',
                        action='store_true',
                        default=False,
                        help='If detections are from an additional class\
                        i.e. another class has already been pre-processed.')



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

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    inference(model_config, eval_config,
              dataset_config, args.base_dir,
              args.ckpt_indices, args.additional_cls)


if __name__ == '__main__':
    tf.app.run()
