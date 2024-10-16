import torch
from torch import nn
from torch.nn import functional as F
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, var='efficientnet_b0', pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        self.encoder = timm.create_model(
            var, pretrained=pretrained, features_only=True
        )

    def forward(self, x):
        features = self.encoder(x)
        return features[-1]

encoder = EfficientNetEncoder()

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(in_channels, num_classes)  
        )

    def forward(self, x):
        return self.fc1(x)
    
