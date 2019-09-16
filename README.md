# TruPercept: Synthetic Data and Trust Modelling for Autonomous Vehicle Cooperative Perception

This repo contains an end-to-end autonomous vehicle cooperative perception model with integrated trust modelling. It also contains testing scripts to be used with the TruPercept dataset located at https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/trupercept.

## Setup
Please install avod (https://github.com/kujason/avod) as well as the submodule wavedata from avod according to the instructions on the avod page. Create an environment using the avod requirements.txt

Clone this repository then add the base tru_percept folder to the python path.

Download our dataset and pretrained avod models (for presil data) from: https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/trupercept

## How to run
cd to the tru_percept directory\
Use the files in the scripts directory to run

You will need to have done the avod preprocessing and have trained avod (or have valid checkpoints and configuration files) if you want to run inference. To run inference for both cars and pedestrians then all tru_percept code:
```
python scripts/run_all.py
```

If using the included predictions then you only need to run the second stage:
```
python scripts/run_stage2.py
```

## Order of Operations:

python run_inference_alt_perspectives.py --checkpoint_name='pyramid_cars_gta' --ckpt_indices=41 --base_dir=/media/bradenhurl/hd/GTAData/TruPercept/object_tru_percept4/ --device='0'

python run_inference_alt_perspectives.py --checkpoint_name='pyramid_people_gta' --ckpt_indices=24 --base_dir=/media/bradenhurl/hd/GTAData/TruPercept/object_tru_percept4/ --device='0' --additional_cls

points_in_3d_box.py
message_evaluations.py
vehicle_trust.py
final_detections.py
eval_utils.py


## Debugging:
The error "Number of samples is less than number of clusters" is due to not having run the AVOD preprocessing scripts before inference. Please follow the steps in the AVOD README for preprocessing mini-batches. This is only required for inference in this project.
Note this same error message could also be due to not having any samples from a particular class. Ensure there is at least one sample from each class that is going to be inferred. This is most likely caused by there not being a 'Cyclist'. Simply add a 'Cyclist' object far away.

There is a bug in avod which does not allow no samples to be in a file. If it crashes during inference in \_filter_labels_by_class then add the index it crashed on to config.INDICES_TO_SKIP

## TODO
- Create settings reader and different preset settings files for experiments
- Probabilistic models
