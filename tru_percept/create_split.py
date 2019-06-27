import os
import sys
import config as cfg

def create_split(base_dir, alt_perspective_dir, split):
    data_dir = alt_perspective_dir + '/label_2/'

    indices = []

    filename = base_dir + '/{}.txt'.format(split)
    count = 0
    with open(filename, 'w+') as f:
        for file in os.listdir(data_dir):
            filepath = data_dir + '/' + file
            idx = int(os.path.splitext(file)[0])
            if idx < cfg.MIN_IDX or idx > cfg.MAX_IDX:
                continue
            if idx in cfg.INDICES_TO_SKIP:
                continue
            printIdx(idx, f)
            count += 1

    # Return true if there are files within the configuration range
    if count > 0:
        return True
    return False
                

def printIdx(idx, f):
    idxStr = "%06d\n" % idx
    f.write(idxStr)