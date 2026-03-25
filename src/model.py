# src/model.py
import torch.nn as nn
from torchvision import models

def build_model():
    # Using pretrained=False is safer across different torchvision versions
    model = models.resnet18(pretrained=False) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model