## av_trust_and_perception

# How to run
cd to the tru_percept directory

Use the files in the scripts directory to run
python run_all.py will run all including inference for both cars and pedestrians
You will need to have done the avod preprocessing and have trained avod (or have valid checkpoints and configuration files) if you want to run inference.

If using the included predictions then you only need to run the second stage:
python scripts/run_stage2.py

# Order of Operations:

python run_inference_alt_perspectives.py --checkpoint_name='pyramid_cars_gta' --ckpt_indices=41 --base_dir=/media/bradenhurl/hd/GTAData/TruPercept/object_tru_percept4/ --device='0'

python run_inference_alt_perspectives.py --checkpoint_name='pyramid_people_gta' --ckpt_indices=24 --base_dir=/media/bradenhurl/hd/GTAData/TruPercept/object_tru_percept4/ --device='0' --additional_cls

points_in_3d_box.py
message_evaluations.py
vehicle_trust.py
final_detections.py
eval_utils.py


Debugging:
The error "Number of samples is less than number of clusters" is due to not having run the AVOD preprocessing scripts before inference. Please follow the steps in the AVOD README for preprocessing mini-batches. This is only required for inference in this project.
Note this same error message could also be due to not having any samples from a particular class. Ensure there is at least one sample from each class that is going to be inferred. This is most likely caused by there not being a 'Cyclist'. Simply add a 'Cyclist' object far away.

There is a bug in avod which does not allow no samples to be in a file. If it crashes during inference in \_filter_labels_by_class then add the index it crashed on to config.INDICES_TO_SKIP

