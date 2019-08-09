import os

import config as cfg
import constants as const
import perspective_utils as p_utils
import vehicle_trust as v_trust

# Find the max index
velo_dir = cfg.DATASET_DIR + '/velodyne'
velo_files = os.listdir(velo_dir)
max_idx = 0
for file in velo_files:
    filepath = velo_dir + '/' + file
    idx = int(os.path.splitext(file)[0])

    if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
        continue

    if idx > max_idx:
        max_idx = idx

# Load the vehicle trust dictionary
v_trust_dict = v_trust.load_vehicle_trust_objs(max_idx)

# Find counts and sum the trust values for malicious/trustworthy vehicles
t_count = 0
t_sum = 0
mal_count = 0
mal_sum = 0
for entity_str in const.valid_perspectives():
    if int(entity_str) in p_utils.FALSE_DETECTIONS:
        mal_count += 1
        mal_sum += v_trust_dict[int(entity_str)].val
    else:
        t_count += 1
        t_sum += v_trust_dict[int(entity_str)].val

# Calculate averages
t_avg = 0
if t_count > 0:
    t_avg = t_sum / t_count

mal_avg = 0
if mal_count > 0:
    mal_avg = mal_sum / mal_count

# Print values
print("Trustworthy count: ", t_count)
print("Trustworthy avg: ", t_avg)
print("Malicious count: ", mal_count)
print("Malicious avg: ", mal_avg)