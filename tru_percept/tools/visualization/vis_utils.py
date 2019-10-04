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
from wavedata.tools.visualization.vtk_text_labels import VtkTextLabels

from wavedata.tools.visualization import vis_utils

import perspective_utils
import trust_utils
import config as cfg
import constants as const
import points_in_3d_boxes

import vtk

# See tru_percept/visualization for scripts with preset options

# OPTIONS (set in calling scripts)
################################################################
'''
img_idx = 7

# Set to true to see predictions (results) from all perspectives
show_results = True

# Sets the perspective ID if alt_persp is true
alt_persp = False
perspID = 61954

# Only uses points within image fulcrum
fulcrum_of_points = True

# Uses the intensity value as colour instead of the image colour
use_intensity = False

# Set to true to view detections from other vehicles
view_received_detections = True
filter_area = False # Filter received detections with perspective area?
receive_from_perspective = -1#61954 # Set to -1 to receive from all perspectives
# Set to only show specific detection
receive_det_id = -1 # Set to -1 to show all detections
only_receive_dets = False # Set to true to only show received dets

# Changes colour of received detections
change_rec_colour = True

# Compare point clouds from two vehicles (for alignment issues)
compare_pcs = False
'''
text_labels = None
text_positions = None
    
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

def visualize(img_idx, show_results, alt_persp, perspID, fulcrum_of_points,
              use_intensity, view_received_detections,
              receive_from_perspective, receive_det_id, only_receive_dets,
              change_rec_colour, compare_pcs, alt_colour_peach=False,
              show_3d_point_count=False, show_orientation=cfg.VISUALIZE_ORIENTATION,
              final_results=False, show_score=False,
              compare_with_gt=False, show_image=True,
              filter_area=cfg.VISUALIZE_AREA_FILTER):
    # Setting Paths
    cam = 2
    dataset_dir = cfg.DATASET_DIR
    print("dataset_dir: ", cfg.DATASET_DIR)

    if img_idx == -1:
        print("Please set the TEST_IDX in the config.py file to see a specific index.")
        img_idx = random.randint(0,101)
        print("Using random index: ", img_idx)

    global text_labels
    global text_positions
    global COLOUR_SCHEME
    if show_3d_point_count or show_score:
        text_labels = []
        text_positions = []

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

    # Load points_in_3d_boxes for each object
    points_dict = points_in_3d_boxes.load_points_in_3d_boxes(img_idx, perspID)

    gt_detections = []
    # Get bounding boxes
    if final_results:
        if filter_area:
            label_dir = os.path.join(dataset_dir, cfg.FINAL_DETS_SUBDIR_AF)
        else:
            label_dir = os.path.join(dataset_dir, cfg.FINAL_DETS_SUBDIR)
        gt_detections = obj_utils.read_labels(label_dir, img_idx, results=show_results)
        if compare_with_gt and not show_results:
            for obj in gt_detections:
                obj.type = "GroundTruth"
        addScoreText(gt_detections, show_3d_point_count, show_score)
    else:
        if (not view_received_detections or receive_from_perspective != -1) and not only_receive_dets:
            gt_detections = perspective_utils.get_detections(dataset_dir, dataset_dir, img_idx, perspID,
                                    perspID, results=show_results, filter_area=filter_area)

            setPointsText(gt_detections, points_dict, show_3d_point_count)
            addScoreTextTrustObjs(gt_detections, show_3d_point_count, show_score)
            gt_detections = trust_utils.strip_objs(gt_detections)
            gt_detections[0].type = "OwnObject"

        if view_received_detections:
            stripped_detections = []
            if receive_from_perspective == -1:
                perspect_detections = perspective_utils.get_all_detections(img_idx, perspID, show_results, filter_area)
                if change_rec_colour:
                    for obj_list in perspect_detections:
                        obj_list[0].obj.type = "OwnObject"
                        if obj_list[0].detector_id == perspID:
                            if compare_with_gt:
                                if obj_list is not None:
                                    for obj in obj_list:
                                        obj.obj.type = "GroundTruth"
                            continue
                        color_str = "Received{:07d}".format(obj_list[0].detector_id)
                        prime_val = obj_list[0].detector_id * 47
                        entity_colour = (prime_val + 13 % 255, (prime_val / 255) % 255, prime_val % 255)
                        COLOUR_SCHEME[color_str] = entity_colour
                        first_obj = True
                        for obj in obj_list:
                            if first_obj:
                                first_obj = False
                                continue
                            obj.obj.type = color_str

                for obj_list in perspect_detections:
                    setPointsText(obj_list, points_dict, show_3d_point_count)
                    addScoreTextTrustObjs(obj_list, show_3d_point_count, show_score)

                stripped_detections = trust_utils.strip_objs_lists(perspect_detections)
            else:
                receive_entity_str = '{:07d}'.format(receive_from_perspective)
                receive_dir = os.path.join(altPerspect_dir, receive_entity_str)
                if os.path.isdir(receive_dir):
                    print("Using detections from: ", receive_dir)
                    perspect_detections = perspective_utils.get_detections(dataset_dir, receive_dir, img_idx, receive_from_perspective,
                                                                            receive_entity_str, results=show_results, filter_area=filter_area)
                    if perspect_detections is not None:
                        color_str = "Received{:07d}".format(receive_from_perspective)
                        prime_val = receive_from_perspective * 47
                        entity_colour = (prime_val + 13 % 255, (prime_val / 255) % 255, prime_val % 255)
                        COLOUR_SCHEME[color_str] = entity_colour
                        first_obj = True
                        for obj in perspect_detections:
                            if first_obj:
                                first_obj = False
                                continue
                            obj.obj.type = color_str
                        setPointsText(perspect_detections, points_dict, show_3d_point_count)
                        addScoreTextTrustObjs(perspect_detections, show_3d_point_count, show_score)
                        stripped_detections = trust_utils.strip_objs(perspect_detections)
                else:
                    print("Could not find directory: ", receive_dir)

            if receive_det_id != -1 and len(stripped_detections) > 0:
                single_det = []
                single_det.append(stripped_detections[receive_det_id])
                stripped_detections = single_det

            if change_rec_colour and alt_colour_peach:
                for obj in stripped_detections:
                    obj.type = "Received"

            if len(stripped_detections) > 0:
                stripped_detections[0].type = "OwnObject"

            if only_receive_dets:
                gt_detections = stripped_detections
                print("Not using main perspective detections")
            else:
                gt_detections = gt_detections + stripped_detections

    if compare_with_gt and show_results:
        label_dir = os.path.join(dataset_dir, cfg.LABEL_DIR)
        real_gt_data = obj_utils.read_labels(label_dir, img_idx, results=False)
        for obj in real_gt_data:
            if obj.type != "DontCare":
                obj.type = "GroundTruth"
        gt_detections = gt_detections + real_gt_data

    visualize_objects_in_pointcloud(gt_detections, COLOUR_SCHEME, dataset_dir,
              img_idx, fulcrum_of_points, use_intensity,
              receive_from_perspective, compare_pcs,
              show_3d_point_count, show_orientation,
              final_results, show_score,
              compare_with_gt, show_image,
              _text_positions=text_positions, _text_labels=text_labels)


# Final visualization function
# Separated out obtaining objects and colour scheme
def visualize_objects_in_pointcloud(objects, COLOUR_SCHEME, dataset_dir,
              img_idx, fulcrum_of_points, use_intensity,
              receive_from_perspective, compare_pcs=False,
              show_3d_point_count=False, show_orientation=cfg.VISUALIZE_ORIENTATION,
              final_results=False, show_score=False,
              compare_with_gt=False, show_image=True,
              _text_positions=None, _text_labels=None):

    image_dir = os.path.join(dataset_dir, 'image_2')
    velo_dir = os.path.join(dataset_dir, 'velodyne')
    calib_dir = os.path.join(dataset_dir, 'calib')

    if compare_pcs:
        fulcrum_of_points = False

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

    if compare_pcs:
        receive_persp_dir = os.path.join(altPerspect_dir, '{:07d}'.format(receive_from_perspective))
        velo_dir2 = os.path.join(receive_persp_dir, 'velodyne')
        print(velo_dir2)
        if not os.path.isdir(velo_dir2):
            print("Error: cannot find velo_dir2: ", velo_dir2)
            exit()
        point_cloud2 = obj_utils.get_lidar_point_cloud(img_idx, calib_dir, velo_dir2,
                                                    im_size=image_shape)
        #Set to true to display point clouds in world coordinates (for debugging)
        display_in_world=False
        if display_in_world:
            point_cloud = perspective_utils.pc_to_world(point_cloud.T, receive_persp_dir, img_idx)
            point_cloud2 = perspective_utils.pc_to_world(point_cloud2.T, dataset_dir, img_idx)
            point_cloud = np.hstack((point_cloud.T, point_cloud2.T))
        else:
            point_cloud2 = perspective_utils.pc_persp_transform(point_cloud2.T, receive_persp_dir, dataset_dir, img_idx)
            point_cloud = np.hstack((point_cloud, point_cloud2.T))

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

    point_colours = None
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

    # Some settings for the initial camera view and point size
    closeView = False
    pitch = 170
    pointSize = 2
    zoom = 1
    if closeView:
        pitch = 180.5
        pointSize = 3
        zoom = 35

    # Create VtkPointCloud for visualization
    vtk_point_cloud = VtkPointCloud()
    if point_colours is not None:
        vtk_point_cloud.set_points(all_points, point_colours)
    else:
        vtk_point_cloud.set_points(all_points)
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(pointSize)

    # Create VtkVoxelGrid for visualization
    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(voxel_grid)

    # Create VtkBoxes for boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(objects, COLOUR_SCHEME, show_orientation)#vtk_boxes.COLOUR_SCHEME_KITTI)

    # Create Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(5, 5, 5)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)
    vtk_renderer.AddActor(axes)
    if _text_positions is not None:
        vtk_text_labels = VtkTextLabels()
        vtk_text_labels.set_text_labels(_text_positions, _text_labels)
        vtk_renderer.AddActor(vtk_text_labels.vtk_actor)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(pitch)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(zoom)

    # Zoom/navigate to the desired camera view then exit
    # Three lines will be output. Paste these here
    # Above forward view
    current_cam.SetPosition(7.512679241328601, -312.20497623371926, -130.38469206536766)
    current_cam.SetViewUp(-0.01952407393317445, -0.44874501090739727, 0.893446543293314)
    current_cam.SetFocalPoint(11.624950999358777, 14.835920755080867, 33.965665867613836)

    # Top down view of synchronization
    current_cam.SetPosition(28.384757950371405, -125.46190537888288, 63.60263366961189)
    current_cam.SetViewUp(-0.02456679343399302, 0.0030507437719906913, 0.9996935358512673)
    current_cam.SetFocalPoint(27.042134804730317, 15.654378427929846, 63.13899801247614)

    current_cam.SetPosition(30.3869590224831, -50.28910856489952, 60.097631136698965)
    current_cam.SetViewUp(-0.0237472244952177, -0.06015048799392083, 0.997906803325274)
    current_cam.SetFocalPoint(27.06695416156647, 15.347824332314035, 63.97499987548391)


    # current_cam.SetPosition(14.391008769593322, -120.06549828061613, -1.567028749253062)
    # current_cam.SetViewUp(-0.02238762832327178, -0.1049057307562059, 0.9942301452644481)
    # current_cam.SetFocalPoint(10.601112314728102, 20.237110061924664, 13.151596441968126)

    # # Top down view of whole detection area
    # current_cam.SetPosition(11.168659642537031, -151.97163016078756, 17.590894639193227)
    # current_cam.SetViewUp(-0.02238762832327178, -0.1049057307562059, 0.9942301452644481)
    # current_cam.SetFocalPoint(6.5828849321501055, 17.79452593368671, 35.400431120570865)

    # Top down view of scenario
    current_cam.SetPosition(2.075612197299923, -76.19063612245675, 5.948366424752178)
    current_cam.SetViewUp(-0.02238762832327178, -0.1049057307562059, 0.9942301452644481)
    current_cam.SetFocalPoint(-0.5129380758134061, 19.637933198314016, 16.00138547483155)

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
    if show_image:
        image = cv2.imread(image_dir + "/%06d.png" % img_idx)
        cv2.imshow("Press any key to continue", image)
        cv2.waitKey()

    # Render in VTK
    vtk_render_window.Render()

    vtk_render_window_interactor.Start()  # Blocking
    # vtk_render_window_interactor.Initialize()   # Non-Blocking  

    # Obtain camera positional information for repeatable views
    print("current_cam.SetPosition{}".format(current_cam.GetPosition()))
    print("current_cam.SetViewUp{}".format(current_cam.GetViewUp()))
    print("current_cam.SetFocalPoint{}".format(current_cam.GetFocalPoint()))

def setPointsText(trust_obj_list,  points_dict, show_3d_point_count):
    if not show_3d_point_count:
        return

    global text_labels
    global text_positions
    for trust_obj in trust_obj_list:
        text_positions.append(trust_obj.obj.t)

        key = trust_obj.detector_id, trust_obj.det_idx
        if key in points_dict:
            points = points_dict[key]
        else:
            points = -1
        
        text_labels.append('p:{}'.format(points))

def addScoreTextTrustObjs(trust_obj_list, show_3d_point_count, show_score):
    obj_list = trust_utils.strip_objs(trust_obj_list)
    addScoreText(obj_list, show_3d_point_count, show_score)

def addScoreText(obj_list, show_3d_point_count, show_score):
    if not show_score:
        return
    global text_labels
    global text_positions

    idx = 0
    if obj_list is not None:
        for obj in obj_list:
            text = ' s:{}'.format(obj.score)
            if not show_3d_point_count:
                text_positions.append(obj.t)
                text_labels.append(text)
            else:
                text_labels[idx] += text

def vis_pc(pc, obj_list, frustum_points=None):

    # Define Fixed Sizes for the voxel grid
    x_min = -85
    x_max = 85
    y_min = -5
    y_max = 5
    z_min = 3
    z_max = 85

    # Comment these out to filter points by area
    x_min = min(pc[0])
    x_max = max(pc[0])
    y_min = min(pc[1])
    y_max = max(pc[1])
    z_min = min(pc[2])
    z_max = max(pc[2])

    # Reshape points into N x [x, y, z]
    all_points = np.array(pc).transpose().reshape((-1, 3))

    # Filter points within certain xyz range
    area_filter = (pc[0] > x_min) & (pc[0] < x_max) & \
                  (pc[1] > y_min) & (pc[1] < y_max) & \
                  (pc[2] > z_min) & (pc[2] < z_max)

    all_points = all_points[area_filter]

    # Create Voxel Grid
    voxel_grid = VoxelGrid()
    voxel_grid_extents = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    print(voxel_grid_extents)

    start_time = time.time()
    voxel_grid.voxelize(all_points, 0.2, voxel_grid_extents)
    end_time = time.time()
    print("Voxelized in {} s".format(end_time - start_time))

    # Some settings for the initial camera view and point size
    closeView = False
    pitch = 170
    pointSize = 4
    zoom = 1
    if closeView:
        pitch = 180.5
        pointSize = 3
        zoom = 35

    # Create VtkPointCloud for visualization
    vtk_point_cloud = VtkPointCloud()
    vtk_point_cloud.set_points(all_points)
    vtk_point_cloud.vtk_actor.GetProperty().SetPointSize(pointSize)

    # Create VtkVoxelGrid for visualization
    vtk_voxel_grid = VtkVoxelGrid()
    vtk_voxel_grid.set_voxels(voxel_grid)

    # Create VtkBoxes for boxes
    vtk_boxes = VtkBoxes()
    vtk_boxes.set_objects(obj_list, COLOUR_SCHEME, False)

    # Create Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(2, 2, 2)

    # Create Voxel Grid Renderer in bottom half
    vtk_renderer = vtk.vtkRenderer()
    vtk_renderer.AddActor(vtk_point_cloud.vtk_actor)
    vtk_renderer.AddActor(vtk_voxel_grid.vtk_actor)
    vtk_renderer.AddActor(vtk_boxes.vtk_actor)
    vtk_renderer.AddActor(axes)
    vtk_renderer.SetBackground(0.2, 0.3, 0.4)


    # Add lines for frustum
    if frustum_points is not None:
        frustum_actor = get_frustum_actor(frustum_points)
        vtk_renderer.AddActor(frustum_actor)

    # Setup Camera
    current_cam = vtk_renderer.GetActiveCamera()
    current_cam.Pitch(pitch)
    current_cam.Roll(180.0)

    # Zooms out to fit all points on screen
    vtk_renderer.ResetCamera()

    # Zoom in slightly
    current_cam.Zoom(zoom)

    # Zoom/navigate to the desired camera view then exit
    # Three lines will be output. Paste these here
    # Above forward view
    current_cam.SetPosition(7.512679241328601, -312.20497623371926, -130.38469206536766)
    current_cam.SetViewUp(-0.01952407393317445, -0.44874501090739727, 0.893446543293314)
    current_cam.SetFocalPoint(11.624950999358777, 14.835920755080867, 33.965665867613836)

    # Reset the clipping range to show all points
    vtk_renderer.ResetCameraClippingRange()

    # Setup Render Window
    vtk_render_window = vtk.vtkRenderWindow()
    vtk_render_window.SetWindowName(
        "Point Cloud and Voxel Grid")
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

    # Render in VTK
    vtk_render_window.Render()

    vtk_render_window_interactor.Start()  # Blocking

def get_frustum_actor(frustum_points):

    # Create frustum with lines
    pts = vtk.vtkPoints()
    origin = [0.0, 0.0, 0.0]
    pts.InsertNextPoint(origin)
    print(len(frustum_points))
    for idx in range(0,len(frustum_points)):
        print("point")
        pts.InsertNextPoint(frustum_points[idx])

    # Setup color
    green = [0, 255, 0]

    # Setup the colors array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    for idx in range(0,8):
        # Add the colors to the colors array
        colors.InsertNextTypedTuple(green)

    # Create the lines

    # Create the first line (between Origin and P0)
    line0 = vtk.vtkLine()
    line0.GetPointIds().SetId(0,0) # the second 0 is the index of the Origin in the vtkPoints
    line0.GetPointIds().SetId(1,1) # the second 1 is the index of P0 in the vtkPoints

    # Create the second line (between Origin and P1)
    line1 = vtk.vtkLine()
    line1.GetPointIds().SetId(0,0) # the second 0 is the index of the Origin in the vtkPoints
    line1.GetPointIds().SetId(1,2) # 2 is the index of P1 in the vtkPoints

    # Next two lines from origin
    line2 = vtk.vtkLine()
    line2.GetPointIds().SetId(0,0)
    line2.GetPointIds().SetId(1,3)
    line3 = vtk.vtkLine()
    line3.GetPointIds().SetId(0,0)
    line3.GetPointIds().SetId(1,4)

    # Create box lines
    line4 = vtk.vtkLine()
    line4.GetPointIds().SetId(0,1)
    line4.GetPointIds().SetId(1,2)
    line5 = vtk.vtkLine()
    line5.GetPointIds().SetId(0,2)
    line5.GetPointIds().SetId(1,3)
    line6 = vtk.vtkLine()
    line6.GetPointIds().SetId(0,3)
    line6.GetPointIds().SetId(1,4)
    line7 = vtk.vtkLine()
    line7.GetPointIds().SetId(0,4)
    line7.GetPointIds().SetId(1,1)

    # Create a cell array to store the lines in and add the lines to it
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(line0)
    lines.InsertNextCell(line1)
    lines.InsertNextCell(line2)
    lines.InsertNextCell(line3)
    lines.InsertNextCell(line4)
    lines.InsertNextCell(line5)
    lines.InsertNextCell(line6)
    lines.InsertNextCell(line7)

    # Create a polydata to store everything in
    linesPolyData = vtk.vtkPolyData()

    # Add the points to the dataset
    linesPolyData.SetPoints(pts)

    # Add the lines to the dataset
    linesPolyData.SetLines(lines)

    # Color the lines - associate the first component (red) of the
    # colors array with the first component of the cell array (line 0)
    # and the second component (green) of the colors array with the
    # second component of the cell array (line 1)
    linesPolyData.GetCellData().SetScalars(colors)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(linesPolyData)
    else:
        mapper.SetInputData(linesPolyData)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor