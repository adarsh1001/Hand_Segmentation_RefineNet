'''
Authors: Adarsh Pal Singh, Paawan Gupta and Ishan Bansal

Classes in VOC 2010- 1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus,
7: car, 8: cat, 9: chair, 10: cow, 11: table, 12: dog, 13: horse, 14: motorbike,
15: person, 16: pottedplant, 17: sheep, 18: sofa, 19: train, 20: tvmonitor}
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow
from matplotlib.colors import LinearSegmentedColormap

def get_map():
    map = {} #Dict

    # 15 corresponds to Person
    map[15] = {} #Dict of Dict
    map[15]['head']       = 0
    map[15]['leye']       = 0                    # left eye
    map[15]['reye']       = 0                   # right eye
    map[15]['lear']       = 0                    # left ear
    map[15]['rear']       = 0                    # right ear
    map[15]['lebrow']     = 0                    # left eyebrow
    map[15]['rebrow']     = 0                    # right eyebrow
    map[15]['nose']       = 0
    map[15]['mouth']      = 0
    map[15]['hair']       = 0
    map[15]['torso']      = 0
    map[15]['neck']       = 0
    map[15]['llarm']      = 1                  # left lower arm
    map[15]['luarm']      = 0                 # left upper arm
    map[15]['lhand']      = 1                   # left hand
    map[15]['rlarm']      = 1                   # right lower arm
    map[15]['ruarm']      = 0                   # right upper arm
    map[15]['rhand']      = 1                   # right hand
    map[15]['llleg']      = 0              	# left lower leg
    map[15]['luleg']      = 0              	# left upper leg
    map[15]['lfoot']      = 0              	# left foot
    map[15]['rlleg']      = 0               	# right lower leg
    map[15]['ruleg']      = 0               	# right upper leg
    map[15]['rfoot']      = 0              	# right foot

    return map

'''
Adapted from Matlab Implementation of color_map for Parts Dataset
'''
def color_map(N=256, normalized=True, matplotlib=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    if matplotlib:
        assert(normalized is True)
        return LinearSegmentedColormap.from_list('VOClabel_cmap', cmap)
    else:
        return cmap
