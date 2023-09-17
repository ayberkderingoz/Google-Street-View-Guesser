import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
import torchvision
from functools import partial
from collections import OrderedDict
import math

import os,inspect,sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)


def convert_relu_to_swish(model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.SiLU(True))
                # setattr(model, child_name, Swish())
            else:
                convert_relu_to_swish(child)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul_(torch.sigmoid(x))
    

class resnet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, dropout_p=0.5):
        super(resnet50, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = torchvision.models.resnet50(pretrained=True)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        # print(modules)
        self.resnet50 = nn.Sequential(*modules)
        convert_relu_to_swish(self.resnet50)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
    def forward(self, x):
        out = self.resnet50(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out
