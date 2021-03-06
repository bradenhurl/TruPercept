#Code modified from: https://github.com/kujason/avod/blob/master/scripts/offline_eval/save_kitti_predictions.py
import sys

import numpy as np
import os
import shutil
import logging
from PIL import Image

from wavedata.tools.core import calib_utils

import avod
from avod.builders import config_builder_util
from avod.builders.dataset_builder import DatasetBuilder
from avod.core import box_3d_projector


def convertPredictionsToKitti(dataset, predictions_root_dir, additional_cls):
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """
    open_mode = 'w+'
    if additional_cls:
        open_mode = 'a+'

    ##############################
    # Options
    ##############################
    global_steps = None
    save_to_base = True
    # global_steps = [28000, 19000, 33000, 34000]

    score_threshold = 0.01

    save_2d = False  # Save 2D predictions
    save_3d = True   # Save 2D and 3D predictions together
    save_alphas = True  # Save alphas (observation angles)

    # Checkpoints below this are skipped
    min_step = 20000

    ##############################
    # End of Options
    ##############################

    final_predictions_root_dir = predictions_root_dir + \
        '/final_predictions_and_scores/' + dataset.data_split

    logging.info('Converting detections from %s', final_predictions_root_dir)

    if not global_steps:
        global_steps = os.listdir(final_predictions_root_dir)
        global_steps.sort(key=int)
        logging.debug('Checkpoints found {}'.format(global_steps))

    for step_idx in range(len(global_steps)):

        global_step = global_steps[step_idx]

        # Skip first checkpoint
        if int(global_step) < min_step:
            continue

        final_predictions_dir = final_predictions_root_dir + \
            '/' + str(global_step)

        if save_to_base:
            kitti_predictions_2d_dir = predictions_root_dir
            kitti_predictions_3d_dir = predictions_root_dir
        else:
            # 2D and 3D prediction directories
            kitti_predictions_2d_dir = predictions_root_dir + \
                '/kitti_predictions_2d/' + \
                dataset.data_split + '/' + \
                str(score_threshold) + '/' + \
                str(global_step) + '/data'
            kitti_predictions_3d_dir = predictions_root_dir + \
                '/kitti_predictions_3d/' + \
                dataset.data_split + '/' + \
                str(score_threshold) + '/' + \
                str(global_step) + '/data'

            if save_2d and not os.path.exists(kitti_predictions_2d_dir):
                os.makedirs(kitti_predictions_2d_dir)
            if save_3d and not os.path.exists(kitti_predictions_3d_dir):
                os.makedirs(kitti_predictions_3d_dir)

        # Do conversion
        num_samples = dataset.num_samples
        num_valid_samples = 0

        logging.info('\nGlobal step: %d', int(global_step))
        logging.info('Converting detections from: %s', final_predictions_dir)

        if save_2d:
            logging.info('2D Detections saved to: %s', kitti_predictions_2d_dir)
        if save_3d:
            logging.info('3D Detections saved to: %s', kitti_predictions_3d_dir)

        for sample_idx in range(num_samples):

            # Print progress
            sys.stdout.write('\rConverting {} / {}'.format(
                sample_idx + 1, num_samples))
            sys.stdout.flush()

            sample_name = dataset.sample_names[sample_idx]

            prediction_file = sample_name + '.txt'

            kitti_predictions_2d_file_path = kitti_predictions_2d_dir + \
                '/' + prediction_file
            kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
                '/' + prediction_file

            predictions_file_path = final_predictions_dir + \
                '/' + prediction_file

            # If no predictions, skip to next file
            if not os.path.exists(predictions_file_path):
                if save_2d:
                    np.savetxt(kitti_predictions_2d_file_path, [])
                if save_3d:
                    np.savetxt(kitti_predictions_3d_file_path, [])
                continue

            all_predictions = np.loadtxt(predictions_file_path, ndmin=2)

            # # Swap l, w for predictions where w > l
            # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
            # fixed_predictions = np.copy(all_predictions)
            # fixed_predictions[swapped_indices, 3] = all_predictions[
            #     swapped_indices, 4]
            # fixed_predictions[swapped_indices, 4] = all_predictions[
            #     swapped_indices, 3]

            score_filter = all_predictions[:, 7] >= score_threshold
            all_predictions = all_predictions[score_filter]

            # If no predictions, skip to next file
            if len(all_predictions) == 0:
                if save_2d:
                    np.savetxt(kitti_predictions_2d_file_path, [])
                if save_3d:
                    np.savetxt(kitti_predictions_3d_file_path, [])
                continue

            # Project to image space
            sample_name = prediction_file.split('.')[0]
            img_idx = int(sample_name)

            # Load image for truncation
            image = Image.open(dataset.get_rgb_image_path(sample_name))

            stereo_calib_p2 = calib_utils.read_calibration(dataset.calib_dir,
                                                           img_idx).p2

            boxes = []
            image_filter = []
            for i in range(len(all_predictions)):
                box_3d = all_predictions[i, 0:7]
                img_box = box_3d_projector.project_to_image_space(
                    box_3d, stereo_calib_p2,
                    truncate=True, image_size=image.size)

                # Skip invalid boxes (outside image space)
                if img_box is None:
                    image_filter.append(False)
                else:
                    image_filter.append(True)
                    boxes.append(img_box)

            boxes = np.asarray(boxes)
            all_predictions = all_predictions[image_filter]

            # If no predictions, skip to next file
            if len(boxes) == 0:
                if save_2d:
                    np.savetxt(kitti_predictions_2d_file_path, [])
                if save_3d:
                    np.savetxt(kitti_predictions_3d_file_path, [])
                continue

            num_valid_samples += 1

            # To keep each value in its appropriate position, an array of zeros
            # (N, 16) is allocated but only values [4:16] are used
            kitti_predictions = np.zeros([len(boxes), 16])

            # Get object types
            all_pred_classes = all_predictions[:, 8].astype(np.int32)
            obj_types = [dataset.classes[class_idx]
                         for class_idx in all_pred_classes]

            # Truncation and Occlusion are always empty (see below)

            # Alpha
            if not save_alphas:
                kitti_predictions[:, 3] = -10 * \
                    np.ones((len(kitti_predictions)), dtype=np.int32)
            else:
                alphas = all_predictions[:, 6] - \
                    np.arctan2(all_predictions[:, 0], all_predictions[:, 2])
                kitti_predictions[:, 3] = alphas

            # 2D predictions
            kitti_predictions[:, 4:8] = boxes[:, 0:4]

            # 3D predictions
            # (l, w, h)
            kitti_predictions[:, 8] = all_predictions[:, 5]
            kitti_predictions[:, 9] = all_predictions[:, 4]
            kitti_predictions[:, 10] = all_predictions[:, 3]
            # (x, y, z)
            kitti_predictions[:, 11:14] = all_predictions[:, 0:3]
            # (ry, score)
            kitti_predictions[:, 14:16] = all_predictions[:, 6:8]

            # Round detections to 3 decimal places
            kitti_predictions = np.round(kitti_predictions, 3)

            # Empty Truncation, Occlusion
            kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                         dtype=np.int32)
            # Empty 3D (x, y, z)
            kitti_empty_2 = -1 * np.ones((len(kitti_predictions), 3),
                                         dtype=np.int32)
            # Empty 3D (h, w, l)
            kitti_empty_3 = -1000 * np.ones((len(kitti_predictions), 3),
                                            dtype=np.int32)
            # Empty 3D (ry)
            kitti_empty_4 = -10 * np.ones((len(kitti_predictions), 1),
                                          dtype=np.int32)

            # Stack 2D predictions text
            kitti_text_2d = np.column_stack([obj_types,
                                             kitti_empty_1,
                                             kitti_predictions[:, 3:8],
                                             kitti_empty_2,
                                             kitti_empty_3,
                                             kitti_empty_4,
                                             kitti_predictions[:, 15]])

            # Stack 3D predictions text
            kitti_text_3d = np.column_stack([obj_types,
                                             kitti_empty_1,
                                             kitti_predictions[:, 3:16]])

            # Save to text files
            if save_2d:
                np.savetxt(kitti_predictions_2d_file_path, kitti_text_2d,
                           newline='\r\n', fmt='%s')
            if save_3d:
                with open(kitti_predictions_3d_file_path, open_mode) as f:
                    np.savetxt(f, kitti_text_3d,
                               newline='\r\n', fmt='%s')

        logging.debug('\nNum valid: %d', num_valid_samples)
        logging.debug('Num samples: %d', num_samples)

    for the_file in os.listdir(predictions_root_dir):
        file_path = os.path.join(predictions_root_dir, the_file)
        try:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logging.debug("Removing folder: %s", file_path)
        except Exception as e:
            print(e)
            logging.exception(e)