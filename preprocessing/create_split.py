import os
import sys

def create_split(base_dir, alt_perspective_dir, split):
    data_dir = alt_perspective_dir + '/label_2/'

    indices = []

    filename = base_dir + '/{}.txt'.format(split)
    with open(filename, 'w+') as f:
        files = os.listdir(data_dir)
        num_files = len(files)
        file_idx = 0
        for file in os.listdir(data_dir):
            filepath = data_dir + '/' + file
            idx = int(os.path.splitext(file)[0])
            if os.stat(filepath).st_size != 0:
                printIdx(idx, f)

            sys.stdout.write("\rWorking on idx: {} / {}".format(
                    file_idx + 1, num_files))
            sys.stdout.flush()
            file_idx = file_idx + 1
                

def printIdx(idx, f):
    idxStr = "%06d\n" % idx
    f.write(idxStr)