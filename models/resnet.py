import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from configs.train_config import NUM_CLASSES


class ResNet(nn.Module):
    def __init__(self,freeze_backbone = True):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        del self.model.fc

        self.model.fc = nn.Linear(512, NUM_CLASSES)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_layer4(self):
        for name, param in self.model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True


    def _forward_impl(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == '__main__':
    model = ResNet()
    image = torch.randn(1, 3, 224, 224)
    output = model(image)
    print(output.shape)
