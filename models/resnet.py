import torch
from torch import nn
from torchvision import models

resnet101_path = '/home/wangshuo/experiment/compet/AIC/models/pretrained_models/resnet101-5d3b4d8f.pth'

resnet152_path = '/home/wangshuo/experiment/compet/AIC/models/pretrained_models/resnet152-b121ed2d.pth'


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

