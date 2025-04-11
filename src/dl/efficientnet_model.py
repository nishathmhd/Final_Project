import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class EfficientNetB3Mammo(nn.Module):
    """Custom EfficientNetB3 for mammogram analysis"""
    
    def __init__(self, num_classes=1, pretrained=True):
        super(EfficientNetB3Mammo, self).__init__()
        
        # Load pretrained EfficientNetB3
        self.base_model = models.efficientnet_b3(pretrained=pretrained)
        
        # Freeze early layers
        for param in self.base_model.features[:5].parameters():
            param.requires_grad = False
            
        # Modify classifier head
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def print_summary(self, input_size=(3, 512, 512)):
        """Print model architecture summary"""
        summary(self, input_size=input_size)

def create_efficientnet_model():
    """Create and return the custom EfficientNetB3 model"""
    model = EfficientNetB3Mammo()
    print("Created custom EfficientNetB3 for mammogram analysis")
    return model

if __name__ == "__main__":
    model = create_efficientnet_model()
    model.print_summary()
