import tru_percept.tru_percept.points_in_3d_boxes as points_3d
import message_evaluations as msg_evals
import vehicle_trust as v_trust
import final_detections as f_dets
import eval_utils
import config as cfg

# Perform remaining tru_percept operations
points_3d.compute_points_in_3d_boxes()
msg_evals.compute_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)