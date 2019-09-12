# This file is used to run multiple aggregation method experiments with the same parameters

import tru_percept.points_in_3d_boxes as points_3d
import message_evaluations as msg_evals
import vehicle_trust as v_trust
import final_detections as f_dets
import eval_utils
import config as cfg

cfg.IOU_MATCHING_THRESHOLD = 0.1
print(cfg.IOU_MATCHING_THRESHOLD)
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)

cfg.IOU_MATCHING_THRESHOLD = 0.3
print(cfg.IOU_MATCHING_THRESHOLD)
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)

cfg.IOU_MATCHING_THRESHOLD = 0.5
print(cfg.IOU_MATCHING_THRESHOLD)
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)

cfg.IOU_MATCHING_THRESHOLD = 0.7
print(cfg.IOU_MATCHING_THRESHOLD)
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)

cfg.IOU_MATCHING_THRESHOLD = 0.9
print(cfg.IOU_MATCHING_THRESHOLD)
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)

eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD, True)