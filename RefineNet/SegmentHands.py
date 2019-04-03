from resnet import rf_lw101
import cv2

import matplotlib
matplotlib.use('agg') #Tk not working on Ada
import matplotlib.pyplot as plt

import numpy as np
import torch
import os

cmap = np.load('cmap.npy')
img_dir = 'images/'
imgs = os.listdir(img_dir)
numC = 7

img_scale = 1./255
img_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

net = rf_lw101(numC, pretrained=True).eval()
net = net.cuda() #Run only with Cuda support!
idx = 1

with torch.no_grad():
    for img in imgs:
        img_path = img_dir + img
        img = np.array(plt.imread(img_path))
        orig_size = img.shape[:2][::-1]

        inter_img = (img*img_scale - img_mean)/img_std

        img_inp = torch.tensor(inter_img.transpose(2, 0, 1)[None]).float()
        img_inp = img_inp.cuda()

        seg = net(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
        seg = cv2.resize(seg, orig_size, interpolation=cv2.INTER_CUBIC)
        seg = cmap[seg.argmax(axis=2).astype(np.uint8)]

        ma = (seg[:,:,0]>=0) & (seg[:,:,0]<=20) & (seg[:,:,2]>=110) & (seg[:,:,2]<=150) & (seg[:,:,1]>=0) & (seg[:,:,1]<=20)
        x = np.zeros(ma.shape)
        x[ma==True] = 255
        x[ma==False] = 0
        cv2.imwrite('segmented_images/'+str(idx)+'.jpg',x)
        idx += 1


