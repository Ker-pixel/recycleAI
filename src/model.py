import torch.nn as nn
from torchvision import models

def build_model():
    # weights=None (modern standard) prevents downloading pretrained ImageNet weights
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model