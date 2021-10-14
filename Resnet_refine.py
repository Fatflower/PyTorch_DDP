# %Header File Start-----------------------------------------------------------
#  Confidential（Unclassified）
#  COPYRIGHT (C) Sun Yat-sen University
#  THIS FILE MAY NOT BE MODIFIED OR REDISTRIBUTED WITHOUT THE
#  EXPRESSED WRITTEN CONSENT OF SYSU
# 
# %-----------------------------------------------------------------------------
#  Title   : Resnet_refine.py
#  Author  : Zhang wentao; 
#  E-mail  : z1282429194@163.com
#  Created : 09/10/2021
#  Description: All Resnet models in this file are based on the models provided 
#               by pytorch and are further designed according to their own needs. 
#               These models can use the parameters of the pre-trained models 
#               provided by Pytorch.
# %-----------------------------------------------------------------------------
#  Modification History:
#  V1.0: 2021.09.10, first created by Zhang wentao
#
# %Header File End--------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class ResNet_refine(nn.Module):
    def __init__(
        self, 
        res: str = 'resnet18',
        pretrained: bool = True,
        classes_num: int = 1000,
    ) -> None:
        super(ResNet_refine, self).__init__()
        if res == 'resnet18':
            bone = models.resnet18(pretrained=pretrained)
            block = 1
        elif res == 'resnet34':
            bone = models.resnet34(pretrained=pretrained)
            block = 1
        elif res == 'resnet50':
            bone = models.resnet50(pretrained=pretrained)
            block = 4
        elif res == 'resnet101':
            bone = models.resnet101(pretrained=pretrained)
            block = 4
        elif res == 'resnet152':
            bone = models.resnet152(pretrained=pretrained)
            block = 4  

        self.feature = nn.Sequential(*list(bone.children())[:-2])
        self.avgpool = (list(bone.children())[-2])
        self.fc = nn.Linear(512*block, classes_num)
        self.first_conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.same = nn.Identity()
        self.feature[0] = self.first_conv
        self.feature[3] = self.same


    def forward(self, x):
        # if x.shape[-1] < 100:
        #     self.feature[0] = self.first_conv
        #     self.feature[3] = self.same


        out = self.feature(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out 

def test_ResNet_refine():
    net = ResNet_refine('resnet18', False, 10)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y)
    summary(net, input_size=(3, 224, 224), device='cpu')
    print(net)


# test model
if __name__ == '__main__':
    test_ResNet_refine()
