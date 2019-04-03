from refinenet import refinenet
import cv2

import matplotlib
matplotlib.use('agg') #Tk not working on Ada
import matplotlib.pyplot as plt

import numpy as np
import torch
import os
import re

cmap = np.load('weights/cmap.npy')
img_dir = './'
numC = 7

img_scale = 1./255
img_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

net = refinenet(numC, pretrained=True).eval()
net = net.cuda() #Run only with Cuda support!
idx = 1

def atoi(strr):
    return int(strr) if strr.isdigit() else strr

def natural_keys(strr):
    return [ atoi(c) for c in re.split('(\d+)', strr) ]

with torch.no_grad():
    video_cap = cv2.VideoCapture('video/video1.mp4')
    flag, image = video_cap.read()
    cnt = 1
    os.chdir('frames/')

    while flag:
        cv2.imwrite('frame%d.jpg' %cnt, image)
        flag, image = video_cap.read()
        cnt += 1

    imgs = os.listdir('.')
    imgs.sort(key=natural_keys)

    for imgname in imgs:
        img = np.array(plt.imread(imgname))
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
        cv2.imwrite('../segmented_frames/'+imgname,x)
        idx += 1
    os.chdir('../segmented_frames')
    frames = os.listdir('.')
    frames.sort(key=natural_keys)
    img = cv2.imread(frames[0])
    h,w,l = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../segmented_video/segmented_video.mp4', fourcc, 24.0, (w, h))
    for frame in frames:
        frame = cv2.imread(frame)
        out.write(frame)
    out.release()


