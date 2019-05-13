import run_inference_alt_perspectives as infer
import message_evaluations as msg_evals
import vehicle_trust as v_trust
import final_detections as f_dets
import eval_utils
import config as cfg

# Note (TODO):
# From tru_percept4
# Removed perspectives 30556, 33031, 692038, 88425, 129917, 138858, 194129, 603986 due to a bug
# All these perspectives are buses or large vehicles with a very high up perspective
# Original data can be found on server

# Create ground planes, splits, infer, and determine # of 3D points in detection boxes
# Can add start_perspective if this fails and want to resume where bug occurred
infer.infer_main('pyramid_cars_gta', [41], False)
infer.infer_main('pyramid_people_gta', [24], True)

# Perform remaining tru_percept operations
msg_evals.compute_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)