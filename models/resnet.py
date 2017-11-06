import torch
from torch import nn
from torchvision import models

from functools import partial
import pickle

resnet101_path = '/home/wangshuo/experiment/compet/AIC/models/pretrained_models/resnet101-5d3b4d8f.pth'

resnet152_path = '/home/wangshuo/experiment/compet/AIC/models/pretrained_models/resnet152-b121ed2d.pth'


resnet50_places365_path = '/home/wangshuo/experiment/compet/AIC/models/pretrained_models/whole_resnet50_places365.pth.tar'


class ResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet, self).__init__()
        model = models.resnet152()
        if pretrained:
            model.load_state_dict(torch.load(resnet152_path))

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        del model.fc
        model.fc = lambda x:x
        self.features = model
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class ResNet_Places365(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_Places365, self).__init__()
        
        # to solve model load error for python 3
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        model = torch.load(resnet50_places365_path)

        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        del model.fc
        model.fc = lambda x:x
        self.features = model
        self.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)
