import os
import numpy as np
import math
import sys

from wavedata.tools.obj_detection import obj_utils

# Script outputs class statistics for difficulty levels

presil_labels = False

dataset_dir = os.path.expanduser('~') + '/Kitti/object/'
max_idx = 7418
split_file = dataset_dir + 'train.txt'

if presil_labels:
    dataset_dir = os.path.expanduser('~') + '/GTAData/object/'
    max_idx = 44000
    split_file = dataset_dir + 'train.txt'

label_dir = dataset_dir + 'training/label_2'

classes = ['Car', 'Pedestrian']
easy = [0,0]
mod = [0,0]
hard = [0,0]
total = [0,0]

def main():
    indices = np.loadtxt(split_file, int)
    for idx in indices:#range(0,max_idx):
        labels = obj_utils.read_labels(label_dir, idx, results=False)

        if labels is not None:
            for obj in labels:
                if obj.type in classes:
                    class_idx = classes.index(obj.type)
                    total[class_idx] += 1
                    # print(obj.occlusion, obj.truncation)
                    if isEasy(obj):
                        easy[class_idx] += 1
                    elif isMod(obj):
                        mod[class_idx] += 1
                    elif isHard(obj):
                        hard[class_idx] += 1
        
        # print progress
        sys.stdout.write('\rReading idx: {} / {}'.format(
            idx + 1, max_idx))
        sys.stdout.flush()

    print("Classes: ", classes)
    print("Total: ", total)
    print("Easy: ", easy)
    print("Mod: ", mod)
    print("Hard: ", hard)


def height(obj):
    return obj.y2 - obj.y1

def occlusion(obj):
    return max(0, obj.occlusion - 0.1)

def truncation(obj):
    return max(0, obj.truncation - 0.1)

def filterObj(obj, min_height, max_truncation, max_occlusion):
    if height(obj) < min_height:
        return False
    if truncation(obj) > max_truncation:
        return False
    if occlusion(obj) > max_occlusion:
        return False
    return True

def isEasy(obj):
    global presil_labels
    if presil_labels:
        return filterObj(obj, 40, 0.15, 0.15)
    return filterObj(obj, 40, 0.15, 0)

def isMod(obj):
    global presil_labels
    if presil_labels:
        return filterObj(obj, 25, 0.30, 0.3)
    return filterObj(obj, 25, 0.30, 1)

def isHard(obj):
    global presil_labels
    if presil_labels:
        return filterObj(obj, 25, 0.5, 0.8)
    return filterObj(obj, 25, 0.5, 2)

main()