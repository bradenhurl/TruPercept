import os


# Constants for x, y, z (standard x,y,z notation)
X = [1., 0., 0.]
Y = [0., 1., 0.]
Z = [0., 0., 1.]


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    tru_percept_root_dir = root_dir()
    return os.path.split(tru_percept_root_dir)[0]