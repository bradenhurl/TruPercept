import config as cfg
import os
from wavedata.tools.obj_detection import obj_utils

# Prints out the entity IDs of large vehicles (i.e. not cars)
# Useful for filtering perspectives from dataset
alt_pers_dir = cfg.DATASET_DIR + '/alt_perspective/'
large_vehicles = []

for entity_str in os.listdir(alt_pers_dir):
    if not os.path.isdir(os.path.join(alt_pers_dir, entity_str)):
        continue
    ego_obj_dir = alt_pers_dir + entity_str + '/ego_object/'
    for f in os.listdir(ego_obj_dir):
        filepath = os.path.join(ego_obj_dir, f)
        idx = int(os.path.splitext(f)[0])
        obj = obj_utils.read_labels(ego_obj_dir, idx)
        if obj[0].type == 'Truck' or obj[0].type == 'Bus':
            large_vehicles.append(int(entity_str))
        break

large_vehicles.sort()
print(large_vehicles)
