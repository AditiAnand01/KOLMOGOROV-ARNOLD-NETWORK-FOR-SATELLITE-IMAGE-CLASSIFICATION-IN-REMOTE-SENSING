import torch
import torch.nn as nn
from torchvision import  models
import gc
import matplotlib.pyplot as plt
import math

from kan import KANLinear

class ConvNeXtKAN(nn.Module):
    def __init__(self):
        super(ConvNeXtKAN, self).__init__()
        # Load pre-trained ConvNeXt model
        self.convnext = models.convnext_tiny(pretrained=True)

        # Freeze ConvNeXt layers 
        for param in self.convnext.parameters():
            param.requires_grad = False

        # Modify the classifier part of ConvNeXt
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()

        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, 10)

    def forward(self, x):
        x = self.convnext(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.kan1(x)
        x = self.kan2(x)
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")
    
from models import ConvNeXt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNeXt().to(device)
# print(model)
print_parameter_details(model)
# summary(model, input_size=(3, 224, 224))