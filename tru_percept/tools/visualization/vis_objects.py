import os
import random

from wavedata.tools.obj_detection import obj_utils

from tools.visualization import vis_utils
import config as cfg
import constants as const

def visualize_objects(objects, img_idx, show_results, alt_persp,
              perspID, fulcrum_of_points=True,
              use_intensity=False, view_received_detections=True,
              receive_from_perspective=-1, only_receive_dets=False,
              compare_pcs=False,
              show_3d_point_count=False, show_orientation=False,
              final_results=False, show_score=False,
              compare_with_gt=True, show_image=True,
              vis_scores=False):

    if cfg.VISUALIZE_AGG_EVALS:
        show_image = False
        
    # Setting Paths
    cam = 2
    dataset_dir = cfg.DATASET_DIR
    print("dataset_dir: ", cfg.DATASET_DIR)

    if img_idx == -1:
        print("Please set the TEST_IDX in the config.py file to see a specific index.")
        img_idx = random.randint(0,101)
        print("Using random index: ", img_idx)

    perspStr = '%07d' % perspID
    altPerspect_dir = os.path.join(dataset_dir,'alt_perspective')
    if alt_persp:
        dataset_dir = dataset_dir + '/alt_perspective/' + perspStr
    else:
        perspID = const.ego_id()

    if show_results:
        label_dir = os.path.join(dataset_dir, 'predictions')
    else:
        label_dir = os.path.join(dataset_dir, 'label_2')

    COLOUR_SCHEME = {
        "Car": (0, 0, 255),  # Blue
        "Pedestrian": (255, 0, 0),  # Red
        "Bus": (0, 0, 255), #Blue
        "Cyclist": (150, 50, 100),  # Purple

        "Van": (255, 150, 150),  # Peach
        "Person_sitting": (150, 200, 255),  # Sky Blue

        "Truck": (0, 0, 255),  # Light Grey
        "Tram": (150, 150, 150),  # Grey
        "Misc": (100, 100, 100),  # Dark Grey
        "DontCare": (255, 255, 255),  # White

        "Received": (255, 150, 150),  # Peach
        "OwnObject": (51, 255, 255),  # Cyan
        "GroundTruth": (0, 255, 0), # Green
    }

    # Load points_in_3d_boxes for each object
    if vis_scores:
        text_positions = []
        text_labels = []
    else:
        text_positions = None
        text_labels = None

    if objects is not None:
        for obj in objects:
            if vis_scores:
                text_positions.append(obj.t)
                txt = '{}'.format(obj.score)
                text_labels.append(txt)


    if compare_with_gt:
        label_dir = os.path.join(dataset_dir, cfg.LABEL_DIR)
        real_gt_data = obj_utils.read_labels(label_dir, img_idx, results=False)
        if real_gt_data is not None:
            for obj in real_gt_data:
                obj.type = "GroundTruth"
            objects = objects + real_gt_data

    vis_utils.visualize_objects_in_pointcloud(objects, COLOUR_SCHEME, dataset_dir,
              img_idx, fulcrum_of_points, use_intensity,
              receive_from_perspective, compare_pcs,
              show_3d_point_count, show_orientation,
              final_results, show_score,
              compare_with_gt, show_image,
              text_positions, text_labels)