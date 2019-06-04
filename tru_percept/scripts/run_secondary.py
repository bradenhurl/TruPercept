import vehicle_trust as v_trust
import final_detections as f_dets
import eval_utils
import config as cfg

# Perform remaining tru_percept operations
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)