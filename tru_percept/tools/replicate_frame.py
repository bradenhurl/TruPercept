import os
from shutil import copyfile

import config as cfg
import constants as const

if cfg.ALLOW_FILE_OVERWRITE == False:
    print("WARNING!!! THIS SCRIPT WILL OVERWRITE SCENE DATA!")
    print("Before running replicate_frame.py ensure you have set the desired options.")
    print("Overwriting files will occur in the directory (and subdirs): ", cfg.DATASET_DIR)

############################### OPTIONS ##################################################
# list of subdirs which will have their files overwritten
# label_aug_2 needs to be copied for data synchronization
subdirs_to_overwrite = [cfg.AVOD_OUTPUT_DIR, 'label_aug_2', 'calib', 'velodyne', 'image_2', 'ego_object', 'position_world']
exts_to_overwrite = ['txt', 'txt', 'txt', 'bin', 'png', 'txt', 'txt',]
idx_to_copy = 0
max_idx = 100


################################ CODE ####################################################
def main():
    # Copy the ego-vehicle dirs
    copy_perspect_dir(cfg.DATASET_DIR)
    # The label dir is sometimes only filtered for the main perspective (used for evaluation)
    dirpath = cfg.DATASET_DIR + '/' + cfg.LABEL_DIR + '/'
    copy_files_in_dir(dirpath, 'txt')

    # Then for all the alternate perspectives
    for entity_str in const.valid_perspectives():
        perspect_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)
        copy_perspect_dir(perspect_dir)
    
def copy_perspect_dir(perspect_dir):
    global subdirs_to_overwrite
    global exts_to_overwrite

    for dir_idx in range(0,len(subdirs_to_overwrite)):
        dirpath = perspect_dir + '/' + subdirs_to_overwrite[dir_idx] + '/'
        copy_files_in_dir(dirpath, exts_to_overwrite[dir_idx])

def copy_files_in_dir(dirpath, file_ext):
    global idx_to_copy
    global max_idx

    src_file = dirpath + '{:06d}.{}'.format(idx_to_copy, file_ext)

    for idx in range(0, max_idx):
        if idx == idx_to_copy:
            continue
        dst_file = dirpath + '{:06d}.{}'.format(idx, file_ext)
        copyfile(src_file, dst_file)
        print("copying file: {} to {}".format(src_file, dst_file))


main()