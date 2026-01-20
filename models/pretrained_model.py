import torch
import torch.nn as nn
from torchvision import models


class PretrainedModel(nn.Module):
    """
    Pretrained ResNet18 model for X-ray Pneumonia Classification.
    Transfer learning giúp model học nhanh hơn và đạt accuracy cao hơn.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Freeze backbone nếu cần (chỉ train classifier)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Thay thế fully connected layer cuối
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone để fine-tune toàn bộ model"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class EfficientNetModel(nn.Module):
    """
    Pretrained EfficientNet-B0 model - nhẹ hơn và hiệu quả hơn ResNet.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Thay classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    # Test model
    model = PretrainedModel(num_classes=2, pretrained=True)
    input_tensor = torch.randn(8, 3, 224, 224)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
