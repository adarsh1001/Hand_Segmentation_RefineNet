import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

#3x3 Conv Block (with Padding)
def conv3x3(inpdepth, outdepth, stride=1, bias=False):
    return nn.Conv2d(inpdepth, outdepth, kernel_size=3, stride=stride, padding=1, bias=bias)

#1x1 Conv Block (Here, Padding is kept as 0)
def conv1x1(inpdepth, outdepth, stride=1, bias=False):
    return nn.Conv2d(inpdepth, outdepth, kernel_size=1, stride=stride, padding=0, bias=bias)

#Chained Residual Pooling Block
class CRPBlock(nn.Module):
    def __init__(self, inpdepth, outdepth, num_stages):
        super(CRPBlock, self).__init__()
        for i in range(num_stages):
            if i == 0:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'), conv1x1(inpdepth, outdepth, stride=1, bias=False))
            else:
                setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'), conv1x1(outdepth, outdepth, stride=1, bias=False))
        self.stride = 1
        self.n_stages = num_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, inp):
        top = inp
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            inp = top + inp
        return inp

stages_suffixes = {0: '_conv', 1: '_conv_relu_varout_dimred'}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inpdepth, outdepth, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inpdepth, outdepth, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv3 = nn.Conv2d(outdepth, outdepth * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outdepth * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inp):
        residual = inp

        interimage = self.conv1(inp)
        interimage = self.bn1(interimage)
        interimage = self.relu(interimage)

        interimage = self.conv2(interimage)
        interimage = self.bn2(interimage)
        interimage = self.relu(interimage)

        interimage = self.conv3(interimage)
        interimage = self.bn3(interimage)

        if self.downsample is not None:
            residual = self.downsample(inp)

        interimage += residual
        out = self.relu(interimage)

        return out


class RefineNetLW(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(RefineNetLW, self).__init__()

        #General Single Layers
        self.drop = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #Blocks made up of multiple layers, Conv->bn->relu->Conv->bn->sum(residual)->relu
        self.layer1 = self.form_Block(block, 64, layers[0])
        self.layer2 = self.form_Block(block, 128, layers[1], stride=2)
        self.layer3 = self.form_Block(block, 256, layers[2], stride=2)
        self.layer4 = self.form_Block(block, 512, layers[3], stride=2)

        #Convolution Layers with 1*1 Kernel Size, Changing input and output depth
        #Blocks of Chain Residual Pooling, Conv->Pool->Sum
        #Adaptive Convolutions
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self.form_CRP(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self.form_CRP(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self.form_CRP(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self.form_CRP(256, 256, 4)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def form_CRP(self, inpdepth, outdepth, stages):
        #Making of the Chain Residual Pooling
        #indepth  : Depth of the input plane
        #outdepth: Depth of the output plane
        #stages    : How many layers of Conv->Pool
        layers = [CRPBlock(inpdepth, outdepth,stages)]
        return nn.Sequential(*layers)

    def form_Block(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inpimage):
        inpimage = self.conv1(inpimage)
        inpimage = self.bn1(inpimage)
        inpimage = self.relu(inpimage)
        inpimage = self.maxpool(inpimage)

        downsample1 = self.layer1(inpimage)
        downsample2 = self.layer2(downsample1)
        downsample3 = self.layer3(downsample2)
        downsample4 = self.layer4(downsample3)

        downsample4 = self.drop(downsample4)
        downsample3 = self.drop(downsample3)

        featuremap4 = self.p_ims1d2_outl1_dimred(downsample4)
        featuremap4 = self.relu(featuremap4)
        featuremap4 = self.mflow_conv_g1_pool(featuremap4)
        featuremap4 = self.mflow_conv_g1_b3_joint_varout_dimred(featuremap4)
        featuremap4 = nn.Upsample(size=downsample3.size()[2:], mode='bilinear', align_corners=True)(featuremap4)

        featuremap3 = self.p_ims1d2_outl2_dimred(downsample3)
        featuremap3 = self.adapt_stage2_b2_joint_varout_dimred(featuremap3)
        featuremap3 = featuremap3 + featuremap4
        featuremap3 = F.relu(featuremap3)
        featuremap3 = self.mflow_conv_g2_pool(featuremap3)
        featuremap3 = self.mflow_conv_g2_b3_joint_varout_dimred(featuremap3)
        featuremap3 = nn.Upsample(size=downsample2.size()[2:], mode='bilinear', align_corners=True)(featuremap3)

        featuremap2 = self.p_ims1d2_outl3_dimred(downsample2)
        featuremap2 = self.adapt_stage3_b2_joint_varout_dimred(featuremap2)
        featuremap2 = featuremap2 + featuremap3
        featuremap2 = F.relu(featuremap2)
        featuremap2 = self.mflow_conv_g3_pool(featuremap2)
        featuremap2 = self.mflow_conv_g3_b3_joint_varout_dimred(featuremap2)
        featuremap2 = nn.Upsample(size=downsample1.size()[2:], mode='bilinear', align_corners=True)(featuremap2)

        featuremap1 = self.p_ims1d2_outl4_dimred(downsample1)
        featuremap1 = self.adapt_stage4_b2_joint_varout_dimred(featuremap1)
        featuremap1 = featuremap1 + featuremap2
        featuremap1 = F.relu(featuremap1)
        featuremap1 = self.mflow_conv_g4_pool(featuremap1)

        out = self.clf_conv(featuremap1)
        return out


def refinenet(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = RefineNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    
    key = 'weights'
    downdict={}
    downdict[key]="weights/hands.pth.tar"
    #model.load_state_dict(maybe_download(key, url), strict=False)
    model.load_state_dict(torch.load(downdict[key], map_location=None), strict=False)
    return model

