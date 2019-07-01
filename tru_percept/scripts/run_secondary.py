import vehicle_trust as v_trust
import message_evaluations as msg_evals
import final_detections as f_dets
import eval_utils
import config as cfg

# Perform remaining tru_percept operations
msg_evals.compute_message_evals()
msg_evals.aggregate_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)