'''
Authors: Adarsh Pal Singh, Paawan Gupta and Ishan Bansal

Parts Dataset for VOC 2010-
http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz

These are basically additional annotations in which the 20 classes of the VOC 2010
Dataset are further segmented into subclasses. We are mostly interested in person-parts
(especially hands). The dataset provides these segmentation information in .mat
files which are not inherent to PyTorch. This code translates the .mat files to
person-parts segments which can be trained using PyTorch's RefineNet.
'''

import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat
from skimage.measure import regionprops
from mappings import color_map
from mappings import get_map
import matplotlib as mpl

MAP = get_map()

def make_label(image_path, anno_path):
    image = plt.imread(image_path) #Read the Image
    image_size = image.shape #Size/Dimensions of Image
    shape = image_size[:-1]
    mat_data = loadmat(anno_path)['anno'][0, 0] #.mat Annotation file
    image_name = mat_data['imname'][0]
    n_objects = mat_data['objects'].shape[1]

    flag = 0

    for obj in mat_data['objects'][0, :]:
        if obj['parts'].shape[1] > 0 and ('lhand' in obj['parts'][0,:]['part_name'] or 'rhand' in obj['parts'][0,:]['part_name']): #We are only interested in Person-Parts!
            flag += 1
            mask = obj['mask']
            props = regionprops(mask)[0]
            class_name = obj['class'][0]
            class_ind = obj['class_ind'][0, 0]
            part_mask = np.zeros(shape, dtype=np.uint8)
            n_parts = obj['parts'].shape[1]

            if n_parts > 0:
                for part in obj['parts'][0, :]:    
                    part_name = part['part_name'][0]
                    pid = MAP[class_ind][part_name]
                    part_mask[part['mask'] > 0] = pid
    if flag != 0:
        return image_size, part_mask
    else:
        return 0, 0


def main():
    #mat_dir = 'Annotations_Part/'
    #origimage_dir = 'JPEGImages/'
    mat_dir = 'Annotations_Part/'
    origimage_dir = 'JPEGImages/'

    label_dir = 'yolabel/'
    personimage_dir = 'yoimage/'

    os.mkdir(label_dir)
    os.mkdir(personimage_dir)

    list_mats = os.listdir(mat_dir)

    for mat in list_mats:
        imname = mat[:-4] + '.jpg'
        image_size, part_mask = make_label(origimage_dir + imname, mat_dir + mat)

        if image_size == 0:
            continue

        copy_cmd = 'cp ' + origimage_dir  + imname + ' ' + personimage_dir
        os.system(copy_cmd)

        mpl.rcParams['savefig.pad_inches'] = 0
        figsize = (image_size[1]*1.0/100, image_size[0]*1.0/100)
        fig = plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
        if np.max(part_mask) == 0:
            ax.imshow(part_mask, cmap='gray')
        else:
            ax.imshow(part_mask, cmap=color_map(N=np.max(part_mask) + 1))
        #plt.savefig(label_dir + imname)
        cv2.imwrite(label_dir + imname, part_mask)
        plt.close()

main()
