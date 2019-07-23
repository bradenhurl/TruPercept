import os
import errno
import shutil
import logging

import config as cfg
import constants as const

def make_dir(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def delete_all_subdirs(subdir):
    delete_subdir(subdir)

    for entity_str in const.valid_perspectives():
        perspect_dir = os.path.join(cfg.ALT_PERSP_DIR, entity_str)
        delete_subdir(subdir, perspect_dir)

def delete_subdir(subdir, basedir=cfg.DATASET_DIR):
    dirpath = os.path.join(basedir, subdir)
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        logging.debug("Deleting directory: {}".format(dirpath))
        shutil.rmtree(dirpath)

def save_objs_to_file(objs, idx, out_dir, results=False):
    out_file = out_dir + '/{:06d}.txt'.format(idx)

    with open(out_file, 'w+') as f:
        if objs is None:
            return
        for obj in objs:
            occ_lvl = 0
            if obj.occlusion > 0.2:
                occ_lvl = 1
            if obj.occlusion > 0.5:
                occ_lvl = 2

            if results:
                kitti_text_3d = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(obj.type,
                    obj.truncation, occ_lvl, obj.alpha, obj.x1, obj.y1, obj.x2,
                    obj.y2, obj.h, obj.w, obj.l, obj.t[0], obj.t[1], obj.t[2], obj.ry, obj.score)
            else:
                kitti_text_3d = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(obj.type,
                    obj.truncation, occ_lvl, obj.alpha, obj.x1, obj.y1, obj.x2,
                    obj.y2, obj.h, obj.w, obj.l, obj.t[0], obj.t[1], obj.t[2], obj.ry)

            f.write('%s\r\n' % kitti_text_3d)