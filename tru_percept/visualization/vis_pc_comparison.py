import tru_percept.tools.visualization.vis_utils as vis_utils

# OPTIONS
################################################################
img_idx = 7

# Set to true to see predictions (results) from all perspectives
show_results = False

# Sets the perspective ID if altPerspective is true
alt_persp = False
perspID = 61954

# Only uses points within image fulcrum
fulcrum_of_points = True
# Uses the intensity value as colour instead of the image colour
use_intensity = False

# Set to true to view detections from other vehicles
view_received_detections = True
receive_from_perspective = 61954 # Set to -1 to receive from all perspectives

# Changes colour of received detections
change_rec_colour = True
alt_colour_peach = True

# Compare point clouds from two vehicles (for alignment issues)
compare_pcs = True

# Set to only show specific detection
receive_det_id = -1 # Set to -1 to show all detections
only_receive_dets = False # Set to true to only show received dets

vis_utils.visualize(img_idx,show_results, alt_persp, perspID, fulcrum_of_points,
                    use_intensity, view_received_detections,
                    receive_from_perspective, receive_det_id, only_receive_dets,
                    change_rec_colour, compare_pcs, alt_colour_peach)

