import os
import random
import time
import cv2
import numpy as np
import wavedata.tools.core.calib_utils as calib
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.core.voxel_grid import VoxelGrid
from wavedata.tools.visualization.vtk_point_cloud import VtkPointCloud
from wavedata.tools.visualization.vtk_voxel_grid import VtkVoxelGrid
from wavedata.tools.visualization.vtk_boxes import VtkBoxes

from wavedata.tools.visualization import vis_utils

import perspective_utils
import trust_utils
import tru_percept.tru_percept.config as cfg

import vtk

# OPTIONS
################################################################
img_idx = 6

# Set to true to see predictions (results) from all perspectives
use_results = False

# Sets the perspective ID if altPerspective is true
altPerspective = False
perspID = 59906

# Only uses points within image fulcrum
fulcrum_of_points = True
# Uses the intensity value as colour instead of the image colour
use_intensity = False

# Set to true to view detections from other vehicles
view_received_detections = True
filter_area = False # Filter received detections with perspective area?

def main():
    # Setting Paths
    cam = 2
    dataset_dir = cfg.DATASET_DIR
    print("dataset_dir: ", cfg.DATASET_DIR)

    ego_id = 40194

    perspStr = '%07d' % perspID
    altPerspect_dir = os.path.join(dataset_dir,'alt_perspective')
    if altPerspective:
        dataset_dir = dataset_dir + '/alt_perspective/' + perspStr

    image_dir = os.path.join(dataset_dir, 'image_2')
    velo_dir = os.path.join(dataset_dir, 'velodyne')
    calib_dir = os.path.join(dataset_dir, 'calib')

    if use_results:
        label_dir = os.path.join(dataset_dir, 'predictions')
    else:
        label_dir = os.path.join(dataset_dir, 'label_2')

    comparePCs = False
    if comparePCs:
        velo_dir2 = os.path.join(dataset_dir, 'velodyne')

    closeView = False
    pitch = 170
    pointSize = 3
    zoom = 1
    if closeView:
        pitch = 180.5
        pointSize = 3
        zoom = 35

    print('=== Loading image: {:06d}.png ==='.format(img_idx))
    print(image_dir)

    image = cv2.imread(image_dir + '/{:06d}.png'.format(img_idx))
    image_shape = (image.shape[1], image.shape[0])

    if use_intensity:
        point_cloud,intensity = obj_utils.get_lidar_point_cloud(img_idx, calib_dir, velo_dir,
                                                    ret_i=use_intensity)
    else:
        point_cloud = obj_utils.get_lidar_point_cloud(img_idx, calib_dir, velo_dir,
                                                    im_size=image_shape)

    if comparePCs:
        point_cloud2 = obj_utils.get_lidar_point_cloud(img_idx, calib_dir, velo_dir2,
                                                    im_size=image_shape)
        point_cloud = np.hstack((point_cloud, point_cloud2))

    # Reshape points into N x [x, y, z]
    all_points = np.array(point_cloud).transpose().reshape((-1, 3))

    # Define Fixed Sizes for the voxel grid
    x_min = -85
    x_max = 85
    y_min = -5
    y_max = 5
    z_min = 3
    z_max = 85

    # Comment these out to filter points by area
    x_min = min(point_cloud[0])
    x_max = max(point_cloud[0])
    y_min = min(point_cloud[1])
    y_max = max(point_cloud[1])
    z_min = min(point_cloud[2])
    z_max = max(point_cloud[2])

    # Filter points within certain xyz range
    area_filter = (point_cloud[0] > x_min) & (point_cloud[0] < x_max) & \
                  (point_cloud[1] > y_min) & (point_cloud[1] < y_max) & \
                  (point_cloud[2] > z_min) & (point_cloud[2] < z_max)

    all_points = all_points[area_filter]

    if fulcrum_of_points:
        # Get point colours
        point_colours = vis_utils.project_img_to_point_cloud(all_points, image,
                                                             calib_dir, img_idx)
    elif use_intensity:
        adjusted = intensity == 65535
        intensity = intensity > 0
        intensity = np.expand_dims(intensity,-1)
        point_colours = np.hstack((intensity*255,intensity*255-adjusted*255,intensity*255-adjusted*255))

    # Create Voxel Grid
    voxel_grid = VoxelGrid()
    voxel_grid_extents = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    print(voxel_grid_extents)

    start_time = time.time()
    voxel_grid.voxelize(all_points, 0.2, voxel_grid_extents)
    end_time = time.time()
    print("Voxelized in {} s".format(end_time - start_time))

    # Get bounding boxes
    gt_detections = obj_utils.read_labels(label_dir, img_idx, results=use_results)
    print(len(gt_detections))

    perspect_detections = perspective_utils.get_all_detections(ego_id, img_idx, perspID, use_results, filter_area)
    stripped_detections = trust_utils.strip_objs_lists(perspect_detections)
    gt_detections = gt_detections + stripped_detections
    # for entity_str in os.listdir(altPerspect_dir):
    #     if os.path.isdir(os.path.join(altPerspect_dir, entity_str)):
    #         perspect_detections = perspective_utils.get_detections(dataset_dir, altPerspect_dir, img_idx, entity_str, results=use_results)
    #         if perspect_detections != None:
    #             if use_results:
    #                 stripped_detections = trust_utils.strip_objs(perspect_detections)
    #                 gt_detections = gt_detections + stripped_detections
    #             else:
    #                 gt_detections = gt_detections + perspect_detections

    # Create VtkPointCloud for visualization
    vtk_point_cloud = VtkPointCloud()
    if fulcrum_of_points or use_intensity:
        vtk_point_cloud.set_points(all_points, point_colours)
    else:
        vtk_point_cloud.set_points(all_points)
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(pointSize)

    # Create VtkVoxelGrid for visualization
    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(voxel_grid)

    COLOUR_SCHEME_PAPER = {
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
    }

    # Create VtkBoxes for boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(gt_detections, COLOUR_SCHEME_PAPER)#vtk_boxes.COLOUR_SCHEME_KITTI)

    # Create Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)
    #vtk_renderer.AddActor(axes)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(pitch)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(zoom)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(
        "Point Cloud and Voxel Grid, Image {}".format(img_idx))
    vtk_render_window.SetSize(1920, 1080)
    vtk_render_window.AddRenderer(vtk_renderer)

    # Setup custom interactor style, which handles mouse and key events
    vtk_render_window_interactor = vtk.vtkRenderWindowInteractor()
    vtk_render_window_interactor.SetRenderWindow(vtk_render_window)

    # Add custom interactor to toggle actor visibilities

    vtk_render_window_interactor.SetInteractorStyle(
        vis_utils.ToggleActorsInteractorStyle([
            vtk_point_cloud.vtk_actor,
            vtk_voxel_grid.vtk_actor,
            vtk_boxes.vtk_actor,
        ]))

    # Show image
    image = cv2.imread(image_dir + "/%06d.png" % img_idx)
    cv2.imshow("Press any key to continue", image)
    cv2.waitKey()

    # Render in VTK
    vtk_render_window.Render()
    vtk_render_window_interactor.Start()  # Blocking
    # renderWindowInteractor.Initialize()   # Non-Blocking

main()