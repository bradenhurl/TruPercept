import run_inference_alt_perspectives as infer
import message_evaluations as msg_evals
import vehicle_trust as v_trust
import final_detections as f_dets
import eval_utils
import config as cfg

#Create ground planes, splits, infer, and determine # of 3D points in detection boxes
print("Step1)")
infer.infer_main('pyramid_cars_gta', [41], False)
infer.infer_main('pyramid_people_gta', [24], True)
print("Step2)")

msg_evals.compute_message_evals()
v_trust.calculate_vehicle_trusts()
f_dets.compute_final_detections()
eval_utils.run_kitti_native_script(cfg.SCORE_THRESHOLD)