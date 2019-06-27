import os
import config as cfg

altPerspect_dir = cfg.DATASET_DIR + '/alt_perspective/'

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

avg = count_imgs / float(p_idx)
print("Total perspectives: ", p_idx)
print("Average frames per perspective: ", avg)
