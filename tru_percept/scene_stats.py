import os
import config as cfg

altPerspect_dir = cfg.DATASET_DIR + '/alt_perspective/'

entity_ids = []

p_idx = 0
count_imgs = 0
perspective_dirs = [ name for name in os.listdir(altPerspect_dir) if os.path.isdir(os.path.join(altPerspect_dir, name)) ]
perspective_dirs.sort(key=float)
p_count = len(perspective_dirs)
for entity_str in perspective_dirs:
    persp_dir = os.path.join(altPerspect_dir, entity_str)
    if not os.path.isdir(persp_dir):
        continue

    p_idx += 1

    persp_img_dir = os.path.join(persp_dir, 'image_2')

    img_list = os.listdir(persp_img_dir)

    count_imgs += len(img_list)

    entity_ids.append([entity_str, len(img_list), min(img_list), max(img_list)])

avg = count_imgs / float(p_idx)
print("Total perspectives: ", p_idx)
print("Average frames per perspective: ", avg)

filepath = cfg.DATASET_DIR + '/scene_stats_entity_frames.txt'
with open(filepath, 'w') as f:
    f.write("frame count, min idx, max idx, entity_id\n")
    for item in entity_ids:
        f.write("{}, {}, {}, {}\n".format(item[1], \
            int(item[2].split('.', 1)[0]), \
            int(item[3].split('.', 1)[0]), item[0]))
print("Entity IDs and valid frame indexes in file: ", filepath)
