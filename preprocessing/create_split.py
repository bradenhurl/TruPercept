import os
import sys

def create_split(base_dir, alt_perspective_dir, split):
    data_dir = alt_perspective_dir + '/label_2/'

    indices = []

    filename = base_dir + '/{}.txt'.format(split)
    with open(filename, 'w+') as f:
        for file in os.listdir(data_dir):
            filepath = data_dir + '/' + file
            idx = int(os.path.splitext(file)[0])
            printIdx(idx, f)
                

def printIdx(idx, f):
    idxStr = "%06d\n" % idx
    f.write(idxStr)