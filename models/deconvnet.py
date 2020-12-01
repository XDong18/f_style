import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from torchvision import models
import torchvision.transforms.functional.resize as resize

vgg16_pretrained = models.vgg16(pretrained=True)
#from .BasicModule import BasicModule
class conv_deconv(nn.Module):

    def __init__(self, class_num):
        super(conv_deconv,self).__init__()
        self.class_num = class_num
        self.conv_features = torch.nn.Sequential(
            # conv1
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv4
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),
            # conv5
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2, return_indices=True),

            # fc6
            torch.nn.Conv2d(512, 4096, 7),
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU(),
            
            #fc7
            torch.nn.Conv2d(4096, 4096, 1),
            torch.nn.BatchNorm2d(4096),
            torch.nn.ReLU())
        
        self.deconv_features = torch.nn.Sequential(
            # deconv6
            torch.nn.ConvTranspose2d(4096, 512, 7),
            torch.nn.BatchNorm2d(512),

            # deconv 5
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            
            # deconv 4
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),

            # deconv 3
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            # deconv 2
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            # deconv 1
            torch.nn.MaxUnpool2d(2, stride=2),
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())

        self.seg_conv = torch.nn.Conv2d(64, self.class_num, 1)


        # self.feature_outputs = [0]*len(self.features)
        self.pool_indices = dict()        
        self.softmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        my_indice = 0
        for i, layer in enumerate(self.conv_features):
            if isinstance(layer, torch.nn.BatchNorm2d):
                my_indice -= 1
            if isinstance(layer, torch.nn.Conv2d):
                if my_indice < len(vgg16_pretrained.features):
                    self.conv_features[i].weight.data = vgg16_pretrained.features[my_indice].weight.data
                    self.conv_features[i].bias.data = vgg16_pretrained.features[my_indice].bias.data
            my_indice += 1

    def forward(self,x):
        origin_size = x.size()[2:]
        output = resize(x, 224)
        index_pool = 1
        for i, layer in enumerate(self.conv_features):
            if isinstance(layer, torch.nn.MaxPool2d):
                output, indices = layer(output)
                # self.feature_outputs[i] = output
                self.pool_indices[index_pool] = indices
                index_pool += 1
            else:
                output = layer(output)
                # self.feature_outputs[i] = output
        # return output
        index_pool -= 1
        for i, layer in enumerate(self.deconv_features):
            if isinstance(layer, torch.nn.MaxUnpool2d):
                output = layer(output, self.pool_indices[index_pool].to(output.device))
                index_pool -= 1
            else:
                output = layer(output)
        out = F.upsample(self.seg_conv(output), origin_size, mode='bilinear', align_corners=True)
        return self.softmax(out)