import torch
import torch.nn as nn
from torchvision import models

class MedicalResNet18(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(MedicalResNet18, self).__init__()
        # Pretrained ResNet18 Backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Replace the FC (fully connected) layer for 7 skin cancer classes
        in_features = self.model.fc.in_features
        setattr(self.model, "fc", nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        ))

    def forward(self, x):
        return self.model(x)

def get_model(num_classes=7, pretrained=True):
    return MedicalResNet18(num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    # Test model shape locally
    net = get_model()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(f"[INFO] Model Output Shape: {out.shape}")  # Output should be [2, 7]