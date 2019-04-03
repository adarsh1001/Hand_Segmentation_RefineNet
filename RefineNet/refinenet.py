import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv1x1(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, input_image):
        top = input_image
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            input_image = top + input_image
        return input_image


stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input_image):
        residual = input_image

        intermediate_image = self.conv1(input_image)
        intermediate_image = self.bn1(intermediate_image)
        intermediate_image = self.relu(intermediate_image)

        intermediate_image = self.conv2(intermediate_image)
        intermediate_image = self.bn2(intermediate_image)

        if self.downsample is not None:
            residual = self.downsample(input_image)

        intermediate_image += residual
        output_image = self.relu(intermediate_image)

        return output_image


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input_image):
        residual = input_image

        intermediate_image = self.conv1(input_image)
        intermediate_image = self.bn1(intermediate_image)
        intermediate_image = self.relu(intermediate_image)

        intermediate_image = self.conv2(intermediate_image)
        intermediate_image = self.bn2(intermediate_image)
        intermediate_image = self.relu(intermediate_image)

        intermediate_image = self.conv3(intermediate_image)
        intermediate_image = self.bn3(intermediate_image)

        if self.downsample is not None:
            residual = self.downsample(input_image)

        intermediate_image += residual
        output_image = self.relu(intermediate_image)

        return output_image


class RefineNet_LW(nn.Module):

    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(RefineNet_LW, self).__init__()
        '''
        layers    : How many layers are to be present in each block
        block     : Decide the change in depth of input and output
        Expansion : 4
        '''

        #General Single Layers
        self.dropout_index = nn.Dropout(p=0.5)
        self.conv1         = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1           = nn.BatchNorm2d(64)
        self.relu          = nn.ReLU(inplace=True)
        self.maxpool       = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #Blocks made up of multiple layers, Conv->bn->relu->Conv->bn->sum(residual)->relu
        self.form_layer64  = self.form_block(block, 64, layers[0])
        self.form_layer128 = self.form_block(block, 128, layers[1], stride=2)
        self.form_layer256 = self.form_block(block, 256, layers[2], stride=2)
        self.form_layer512 = self.form_block(block, 512, layers[3], stride=2)
        
        #Convolution Layers with 1*1 Kernel Size, Changing input and output depth
        self.convert_2048_512 = conv1x1(2048, 512, bias=False)
        self.convert_1024_256 = conv1x1(1024, 256, bias=False)
        self.convert_512_256  = conv1x1(512, 256, bias=False)
        self.convert_256_256  = conv1x1(256, 256, bias=False)

        #Blocks of Chain Residual Pooling, Conv->Pool->Sum 
        self.conv_pool_1 = self.form_CRP(512, 512, 4)
        self.conv_pool_2 = self.form_CRP(256, 256, 4)
        self.conv_pool_3 = self.form_CRP(256, 256, 4)
        self.conv_pool_4 = self.form_CRP(256, 256, 4)

        self.mflow_conv_g1 = conv1x1(512, 256, bias=False)
        self.mflow_conv_g2 = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3 = conv1x1(256, 256, bias=False)
        
        #Adaptive Convolutions
        self.adaptiveConv_stage2 = conv1x1(256, 256, bias=False)
        self.adaptiveConv_stage3 = conv1x1(256, 256, bias=False)
        self.adaptiveConv_stage4 = conv1x1(256, 256, bias=False)

        self.final_classifier = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

    def form_CRP(self, in_planes, out_planes, stages):
        #Making of the Chain Residual Pooling
        #in_plane  : Depth of the input plane
        #out_planes: Depth of the output plane
        #stages    : How many layers of Conv->Pool
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

    def form_block(self, block, planes, blocks, stride=1):
        #Making Block Containing Conv->bn->relu
        
        #Check if their is difference between input and output depth
        #and form the batch accordingly
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input_image):
        input_image = self.conv1(input_image)
        input_image = self.bn1(input_image)
        input_image = self.relu(input_image)
        input_image = self.maxpool(input_image)

        downsample1 = self.form_layer64(input_image)
        downsample2 = self.form_layer128(downsample1)
        downsample3 = self.form_layer256(downsample2)
        downsample4 = self.form_layer512(downsample3)

        downsample4 = self.dropout_index(downsample4)
        downsample3 = self.dropout_index(downsample3)

        feature_map_4 = self.convert_2048_512(downsample4)
        feature_map_4 = self.relu(feature_map_4)
        feature_map_4 = self.conv_pool_1(feature_map_4)
        feature_map_4 = self.mflow_conv_g1(feature_map_4)
        feature_map_4 = nn.Upsample(size=downsample3.size()[2:], mode='bilinear', align_corners=True)(feature_map_4)

        feature_map_3 = self.convert_1024_256(downsample3)
        feature_map_3 = self.adaptiveConv_stage2(feature_map_3)
        feature_map_3 = feature_map_3 + feature_map_4
        feature_map_3 = F.relu(feature_map_3)
        feature_map_3 = self.conv_pool_2(feature_map_3)
        feature_map_3 = self.mflow_conv_g2(feature_map_3)
        feature_map_3 = nn.Upsample(size=downsample2.size()[2:], mode='bilinear', align_corners=True)(feature_map_3)

        feature_map_2 = self.convert_512_256(downsample2)
        feature_map_2 = self.adaptiveConv_stage3(feature_map_2)
        feature_map_2 = feature_map_2 + feature_map_3
        feature_map_2 = F.relu(feature_map_2)
        feature_map_2 = self.conv_pool_3(feature_map_2)
        feature_map_2 = self.mflow_conv_g3(feature_map_2)
        feature_map_2 = nn.Upsample(size=downsample1.size()[2:], mode='bilinear', align_corners=True)(feature_map_2)

        feature_map_1 = self.convert_256_256(downsample1)
        feature_map_1 = self.adaptiveConv_stage4(feature_map_1)
        feature_map_1 = feature_map_1 + feature_map_2
        feature_map_1 = F.relu(feature_map_1)
        feature_map_1 = self.conv_pool_4(feature_map_1)

        output_image = self.final_classifier(feature_map_1)
        return output_image


def refinenet(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = RefineNet_LW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    
    key = 'weights'
    downdict={}
    downdict[key]="weights/hands.pth.tar"
    model.load_state_dict(torch.load(downdict[key], map_location=None), strict=False)
    return model

