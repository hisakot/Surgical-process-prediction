import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

import common

model_urls = {
        "resnet50" : "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        }

class ResNet50(nn.Module):
    def __init__(self, pretrained, num_input_channel, num_output):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True, num_classes=1000)
        self.resnet50.conv1 = nn.Conv2d(num_input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_output)
    
    def forward(self, x):
        x = self.resnet50(x.float())
        return x

