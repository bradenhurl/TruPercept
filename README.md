# TruPercept: Synthetic Data and Trust Modelling for Autonomous Vehicle Cooperative Perception

This repo contains an end-to-end autonomous vehicle cooperative perception model with integrated trust modelling. It also contains testing scripts to be used with the TruPercept dataset located at https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/trupercept.

If you use the data or code in your academic work, please consider citing our paper: https://arxiv.org/abs/1909.07867

## Setup
Please install avod (https://github.com/kujason/avod) as well as the submodule wavedata from avod according to the instructions on the avod page. Create an environment using the avod requirements.txt

Clone this repository then add the base tru_percept folder to the python path.

Download our dataset and pretrained avod models (for presil data) from: https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/projects/trupercept

Move the four .py files in folder "other_vtk_tools" to your own "wavedata/tools/visualizaton" path

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

Modify file path on lines 17 and 18 of the file "run_all.py", which should correspond to the folder name under "avod/data/outputs" 

**Other modification about path :**

1) File "config.py", line 7 , correspond to the downloaded dataset ”trupercept1/training“

2. File "tru_percept/tools/print_difficulty_stats.py", line17 , modify '/GTAData/object/' to your own path of dataset 'XXX/XXX/trupercept1/'
3. Same as the operation in 2 above , file "vtk_vis_perspectives.py", line30


## Debugging:
The error "Number of samples is less than number of clusters" is due to not having run the AVOD preprocessing scripts before inference. Please follow the steps in the AVOD README for preprocessing mini-batches. This is only required for inference in this project.
Note this same error message could also be due to not having any samples from a particular class. Ensure there is at least one sample from each class that is going to be inferred. This is most likely caused by there not being a 'Cyclist'. Simply add a 'Cyclist' object far away.

There is a bug in avod which does not allow no samples to be in a file. If it crashes during inference in \_filter_labels_by_class then add the index it crashed on to config.INDICES_TO_SKIP

Modify the file "config.py", line 48, when you use 'True' would generate Error alert

Modify the file "constants.py", line 21 ; "correct_synchronization.py", line 114/130/131. The use of function: `obj_utils.read_labels` has three parameters, you can modify the arguments according to your needself

In path "wavedata/tools/obj_detection/obj_utils", add 'self.id' and 'self.speed' in function:`__init__(self)` 

## Suggestion:

If your terminal return an Error about `nan_mask` and `point_clouds`, you can modify on the file "points_in_3d_boxes.py", comment the code on line 175, thus line 176 should be `point_cloud = all_points.T` 

## TODO
- Create settings reader and different preset settings files for experiments
- Probabilistic models



## Other options:

You can use Anaconda3 to create a virtual environment. All codes from avod to TruPercept can be run in this way. 

If you use Anaconda3, run `source ~/.bashrc`, add your dynamic lib path,such as `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xxxxx/anaconda3/pkgs/libboost-1.73.0-h3ff78a5_11/lib` to the end of the file 

