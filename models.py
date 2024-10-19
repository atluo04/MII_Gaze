import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetEncoder, self).__init__()
        if pretrained:
            self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.encoder = efficientnet_b0()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

    def forward(self, x):
        features = self.encoder(x)
        return features

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),  
            nn.Linear(in_channels, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc1(x)

class BaselineModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(BaselineModel, self).__init__()
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        self.classification_head = ClassificationHead(
            in_channels=1280,
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.classification_head(features)
        return output


