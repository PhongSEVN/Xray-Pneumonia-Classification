import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self._make_layer(3,32,3,1,1)
        self.conv2 = self._make_layer(32,64,3,1,1)
        self.conv3 = self._make_layer(64,128,3,1,1)
        self.conv4 = self._make_layer(128,256,3,1,1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def _make_layer(self, inchanel, outchanel, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inchanel, out_channels=outchanel, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=outchanel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.gap(x)

        x = self.flatten(x)

        x = self.linear(x)
        return x

if __name__ == '__main__':
    model = CNN_model()
    input = torch.randn(8, 3, 224, 224)
    output = model(input)
    print(output.shape)